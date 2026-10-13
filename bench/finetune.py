"""LoRA fine-tuning on the synthetic Camunda pairs (results/train.jsonl) on CUDA, Apple MPS or CPU.

    uv run python bench/finetune.py <hf-model> <adapter-dir> [epochs, default 1]

Each example is the same prompt the benchmark uses (retrieved context + question); the loss covers only the answer.
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from corpus import tables_of
from eval_llm import contexts, messages, prompt
from hardware import pick

EMBEDDER = "intfloat/multilingual-e5-small"
TRAIN = Path(__file__).parent / "results" / "train.jsonl"
ACCUM = 8  # examples per optimizer step
LR = 2e-4
SEQ_BUCKET = 128  # sequence lengths are padded to a multiple of this
ANSWER_BUCKET = 64  # likewise the number of positions the loss is computed on


def examples(name: str, tok) -> list[tuple[list[int], int]]:
    """(token ids, prompt length) per training pair."""
    pairs = [json.loads(line) for line in TRAIN.read_text().splitlines()]
    for p in pairs:
        p["tables"] = tables_of(p["sql"])
    out = []
    for p, ctx in zip(pairs, contexts(EMBEDDER, pairs)):
        text, stop = prompt(name, tok, ctx, p["q"])
        if stop:
            full = text + p["sql"] + stop + tok.eos_token
        else:
            answer = {"role": "assistant", "content": f"```sql\n{p['sql']}\n```"}
            full = tok.apply_chat_template(messages(ctx, p["q"]) + [answer], tokenize=False)
            assert full.startswith(text)
        special = bool(stop)
        n_prompt = len(tok(text, add_special_tokens=special)["input_ids"])
        out.append((tok(full, add_special_tokens=special)["input_ids"], n_prompt))
    return out


def main() -> None:
    name, out_dir = sys.argv[1], Path(sys.argv[2])
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    device, dtype = pick()
    tok = AutoTokenizer.from_pretrained(name)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    data = examples(name, tok)
    print(f"{len(data)} examples, max {max(len(ids) for ids, _ in data)} tokens; device {device}, {dtype}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear",
                                             task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    total = math.ceil(len(data) * epochs / ACCUM)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * 0.5 * (1 + math.cos(math.pi * s / total)))

    rnd = random.Random(0)
    model.train()
    step, seen, running, t0 = 0, 0, 0.0, time.perf_counter()
    for epoch in range(epochs):
        rnd.shuffle(data)
        for ids, n_prompt in data:
            # MPS compiles a graph per tensor shape, so lengths are rounded up to buckets: left padding (masked,
            # positions start after it) keeps the answer at the end of the sequence.
            pad = -len(ids) % SEQ_BUCKET
            x = torch.tensor([[pad_id] * pad + ids], device=device)
            mask = torch.tensor([[0] * pad + [1] * len(ids)], device=device)
            positions = torch.tensor([[0] * pad + list(range(len(ids)))], device=device)
            # Logits only around the answer: over the whole prompt they would take gigabytes
            # (sequence length x 150k vocabulary) for positions the loss ignores anyway.
            n_answer = len(ids) - n_prompt
            keep = n_answer + 1 + (-(n_answer + 1) % ANSWER_BUCKET)
            logits = model(input_ids=x, attention_mask=mask, position_ids=positions, logits_to_keep=keep).logits[0, :-1]
            targets = x[0, -(keep - 1):].clone()
            targets[: keep - 1 - n_answer] = -100
            loss = torch.nn.functional.cross_entropy(logits.float(), targets, ignore_index=-100)
            (loss / ACCUM).backward()
            running += loss.item()
            seen += 1
            if seen % ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                if step % 5 == 0:
                    print(f"epoch {epoch + 1} step {step}/{total} loss {running / (5 * ACCUM):.4f} "
                          f"{time.perf_counter() - t0:.0f}s", flush=True)
                    running = 0.0
    model.save_pretrained(out_dir)
    print(f"adapter -> {out_dir} in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
