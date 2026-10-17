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
    model: Any, tok: Any, texts: list[str], batch_size: int = 32, max_new_tokens: int = 12
) -> tuple[list[str | None], list[float]]:
    """Greedy decoding with a KV cache, keeping only last-position logits per step.

    Returns (label or None if outside the enum, joint probability of the generated tokens incl. <|im_end|>).
    Memory stays at batch x vocab per step (HF ``output_scores`` would hold batch x steps x vocab in fp32).
    Labels are at most 9 tokens, so 12 new tokens always suffice.
    """
    model.eval()
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    preds: list[str | None] = []
    confs: list[float] = []
    for i in range(0, len(texts), batch_size):
        chunk = [prompt_ids(tok, t) for t in texts[i : i + batch_size]]
        n = max(len(c) for c in chunk)
        ids = torch.full((len(chunk), n), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(chunk), n), dtype=torch.long)
        for j, c in enumerate(chunk):  # left padding
            ids[j, n - len(c) :] = torch.tensor(c)
            att[j, n - len(c) :] = 1
        ids, att = ids.cuda(), att.cuda()
        pos = (att.cumsum(-1) - 1).clamp(min=0)
        b = ids.shape[0]
        out_tokens = torch.zeros((b, 0), dtype=torch.long, device="cuda")
        logp = torch.zeros(b, device="cuda")
        done = torch.zeros(b, dtype=torch.bool, device="cuda")
        past = None
        step_ids, step_pos = ids, pos
        for _ in range(max_new_tokens):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(
                    input_ids=step_ids,
                    attention_mask=att,
                    position_ids=step_pos,
                    past_key_values=past,
                    use_cache=True,
                    logits_to_keep=1,
                )
            past = out.past_key_values
            lp = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)
            nxt = lp.argmax(dim=-1)
            logp += torch.where(done, torch.zeros_like(logp), lp.gather(1, nxt[:, None]).squeeze(1))
            nxt = torch.where(done, torch.full_like(nxt, eos), nxt)
            out_tokens = torch.cat([out_tokens, nxt[:, None]], dim=1)
            done |= nxt == eos
            if bool(done.all()):
                break
            att = torch.cat([att, torch.ones((b, 1), dtype=att.dtype, device="cuda")], dim=1)
            step_pos = step_pos[:, -1:] + 1
            step_ids = nxt[:, None]
        for j in range(b):
            toks = out_tokens[j].tolist()
            k = toks.index(eos) if eos in toks else len(toks)
            preds.append(canonical_label(tok.decode(toks[:k], skip_special_tokens=True)))
            confs.append(float(torch.exp(logp[j])) if eos in toks else 0.0)
        del past
    return preds, confs
