"""QLoRA/LoRA fine-tuning of a Qwen2.5 student on teacher labels (HF transformers + PEFT + bitsandbytes).

  .venv/Scripts/python.exe scripts/train.py configs/train/q05_r8.yaml

Logs per step (loss, LR) and per epoch (val loss, val accuracy/macro-F1 vs teacher labels and vs gold,
sample outputs) to W&B in OFFLINE mode and to results/train/<name>/*.csv|json. Saves the LoRA adapter per
epoch under artifacts/ (gitignored) and records the best epoch by val accuracy vs teacher labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from student_hf import generate_labels, load_tokenizer, prompt_ids  # noqa: E402

from distill.data import read_jsonl  # noqa: E402
from distill.metrics import accuracy, macro_f1  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_examples(rows: list[dict[str, Any]], tok: Any, max_len: int) -> list[dict[str, list[int]]]:
    out = []
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    for r in rows:
        p = prompt_ids(tok, str(r["text"]))
        c = tok(str(r["teacher"]), add_special_tokens=False)["input_ids"] + [eos]
        ids = (p + c)[:max_len]
        labels = ([-100] * len(p) + c)[:max_len]
        out.append({"input_ids": ids, "labels": labels})
    return out


def collate(batch: list[dict[str, list[int]]], pad_id: int) -> dict[str, torch.Tensor]:
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), n), -100, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        k = len(b["input_ids"])
        ids[i, :k] = torch.tensor(b["input_ids"])
        lab[i, :k] = torch.tensor(b["labels"])
        att[i, :k] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


@torch.no_grad()
def val_loss(model: Any, examples: list[dict[str, list[int]]], pad_id: int, bs: int) -> float:
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(examples), bs):
        b = {k: v.cuda() for k, v in collate(examples[i : i + bs], pad_id).items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(**b)
        k = int((b["labels"] != -100).sum())
        tot += float(out.loss) * k
        n += k
    return tot / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--max-train", type=int, default=None, help="debug: cap training examples")
    ap.add_argument("--data-dir", default=str(ROOT / "data/processed"))
    ap.add_argument("--smoke", action="store_true", help="debug run: outputs go to artifacts/smoke, no W&B")
    a = ap.parse_args()
    cfg: dict[str, Any] = yaml.safe_load(Path(a.config).read_text())
    name = cfg["name"]
    tcfg, lcfg = cfg["train"], cfg["lora"]
    set_seed(int(tcfg["seed"]))

    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_DIR", str(ROOT))
    import wandb
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, get_cosine_schedule_with_warmup

    if a.smoke:
        os.environ["WANDB_MODE"] = "disabled"
    out_dir = (ROOT / "artifacts" / "smoke" if a.smoke else ROOT / "results" / "train") / name
    art_dir = ROOT / "artifacts" / "adapters" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_jsonl(Path(a.data_dir) / "train.jsonl")[: a.max_train or None]
    val_rows = read_jsonl(Path(a.data_dir) / "val.jsonl")

    tok = load_tokenizer(cfg["base_model"], cfg.get("revision"))
    quant = None
    if cfg.get("quantization") == "nf4":
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        revision=cfg.get("revision"),
        quantization_config=quant,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    if quant is not None:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=bool(tcfg.get("grad_ckpt", False))
        )
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(lcfg["r"]),
            lora_alpha=int(lcfg["alpha"]),
            lora_dropout=float(lcfg["dropout"]),
            target_modules=list(lcfg["target_modules"]),
            task_type="CAUSAL_LM",
        ),
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    train_ex = build_examples(train_rows, tok, int(tcfg["max_len"]))
    val_ex = build_examples(val_rows, tok, int(tcfg["max_len"]))
    bs, accum, epochs = int(tcfg["batch_size"]), int(tcfg.get("grad_accum", 1)), int(tcfg["epochs"])
    steps_per_epoch = math.ceil(len(train_ex) / (bs * accum))
    total_steps = steps_per_epoch * epochs
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg["weight_decay"]),
    )
    sched = get_cosine_schedule_with_warmup(opt, int(total_steps * float(tcfg["warmup_ratio"])), total_steps)

    run = wandb.init(
        project="model-distillation-pipeline",
        name=name,
        config={
            **cfg,
            "n_train": len(train_ex),
            "n_val": len(val_ex),
            "trainable_params": trainable,
            "total_params": total,
        },
    )
    step_f = (out_dir / "steps.csv").open("w", newline="")
    step_w = csv.writer(step_f)
    step_w.writerow(["step", "epoch", "loss", "lr"])
    epochs_log: list[dict[str, Any]] = []
    samples_log: list[dict[str, Any]] = []
    pad_id = tok.pad_token_id
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    step = 0
    rng = random.Random(int(tcfg["seed"]))
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(train_ex)))
        rng.shuffle(order)
        for i in range(0, len(order), bs * accum):
            opt.zero_grad(set_to_none=True)
            loss_acc = 0.0
            micro = [order[j : j + bs] for j in range(i, min(i + bs * accum, len(order)), bs)]
            for mb in micro:
                b = {k: v.cuda() for k, v in collate([train_ex[j] for j in mb], pad_id).items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(**b).loss / len(micro)
                loss.backward()
                loss_acc += loss.item()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            step += 1
            lr = sched.get_last_lr()[0]
            step_w.writerow([step, round(step / steps_per_epoch, 4), round(loss_acc, 5), lr])
            run.log(
                {"train/loss": loss_acc, "train/lr": lr, "train/epoch": step / steps_per_epoch}, step=step
            )
        # ---- per-epoch validation ----
        vl = val_loss(model, val_ex, pad_id, 16)
        preds, confs = generate_labels(model, tok, [str(r["text"]) for r in val_rows], batch_size=32)
        t_lab = [str(r["teacher"]) for r in val_rows]
        g_lab = [str(r["gold"]) for r in val_rows]
        row = {
            "epoch": epoch,
            "step": step,
            "val_loss": vl,
            "val_acc_vs_teacher": accuracy(t_lab, preds),
            "val_f1_vs_teacher": macro_f1(t_lab, preds),
            "val_acc_vs_gold": accuracy(g_lab, preds),
            "val_invalid_rate": sum(p is None for p in preds) / len(preds),
            "elapsed_s": round(time.time() - t_start, 1),
        }
        epochs_log.append(row)
        run.log({f"val/{k}": v for k, v in row.items() if k not in ("epoch", "step")}, step=step)
        samples = [
            {
                "epoch": epoch,
                "text": r["text"],
                "teacher": r["teacher"],
                "gold": r["gold"],
                "student": p,
                "conf": round(c, 4),
            }
            for r, p, c in list(zip(val_rows, preds, confs, strict=True))[:8]
        ]
        samples_log.extend(samples)
        run.log(
            {"val/samples": wandb.Table(columns=list(samples[0]), data=[list(s.values()) for s in samples])},
            step=step,
        )
        model.save_pretrained(art_dir / f"epoch-{epoch}")
        print(json.dumps(row), flush=True)
    step_f.close()
    best = max(epochs_log, key=lambda r: (r["val_acc_vs_teacher"], -r["val_loss"]))
    summary = {
        "name": name,
        "config": cfg,
        "n_train": len(train_ex),
        "n_val": len(val_ex),
        "trainable_params": trainable,
        "total_params": total,
        "train_time_s": round(time.time() - t_start, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "best_epoch": best["epoch"],
        "best": best,
        "epochs": epochs_log,
        "gpu": torch.cuda.get_device_name(0),
    }
    with (out_dir / "epochs.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(epochs_log[0]))
        w.writeheader()
        w.writerows(epochs_log)
    (out_dir / "samples.json").write_text(json.dumps(samples_log, indent=1) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    run.summary.update({"best_epoch": best["epoch"], **{f"best/{k}": v for k, v in best.items()}})
    run.finish()
    print(json.dumps({k: summary[k] for k in ("name", "best_epoch", "train_time_s", "peak_vram_gb")}))


if __name__ == "__main__":
    main()
