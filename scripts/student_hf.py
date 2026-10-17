"""HF-transformers helpers for the student: prompt building (identical to serving), batched greedy
generation with a per-example confidence (joint probability of the generated label tokens)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from distill.labels import canonical_label  # noqa: E402
from distill.student import student_messages  # noqa: E402


def load_tokenizer(base: str, revision: str | None = None) -> Any:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base, revision=revision)  # nosec B615 - revision pinned in config
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def prompt_ids(tok: Any, text: str) -> list[int]:
    ids: list[int] = tok.apply_chat_template(
        student_messages(text), add_generation_prompt=True, tokenize=True
    )
    return ids


@torch.no_grad()
def generate_labels(
    model: Any, tok: Any, texts: list[str], batch_size: int = 64, max_new_tokens: int = 24
) -> tuple[list[str | None], list[float]]:
    model.eval()
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    preds: list[str | None] = []
    confs: list[float] = []
    for i in range(0, len(texts), batch_size):
        chunk = [prompt_ids(tok, t) for t in texts[i : i + batch_size]]
        n = max(len(c) for c in chunk)
        ids = torch.full((len(chunk), n), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(chunk), n), dtype=torch.long)
        for j, c in enumerate(chunk):  # left padding for generation
            ids[j, n - len(c) :] = torch.tensor(c)
            att[j, n - len(c) :] = 1
        ids, att = ids.cuda(), att.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.generate(
                input_ids=ids,
                attention_mask=att,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=eos,
                pad_token_id=tok.pad_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )
        gen = out.sequences[:, n:]
        scores = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
        for j in range(gen.shape[0]):
            toks = gen[j].tolist()
            k = toks.index(eos) if eos in toks else len(toks)
            text = tok.decode(toks[:k], skip_special_tokens=True)
            lp = float(scores[j, :k].float().sum()) if k > 0 else -1e9
            preds.append(canonical_label(text))
            confs.append(float(torch.exp(torch.tensor(lp))))
    return preds, confs
