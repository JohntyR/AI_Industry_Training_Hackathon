"""
LoRA SFT for the synthesis role. Runs INSIDE the NeMo container (torch 2.8 + peft 0.18).

Uses plain HF Trainer + peft rather than a NeMo recipe: the organizer scaffold
(~/Cognitivo_Training/finagent-finetune) is not present on this node, and a known-good
script beats reverse-engineering a missing one under time pressure.

Trains on COMPLETION TOKENS ONLY (the prompt is masked out) so the model learns to produce
the grounded answer, not to reproduce the tool digest.

All paths/hyperparameters come from env — see training/node_env.sh.
"""
import json, os, sys

import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments)
from peft import LoraConfig, get_peft_model

MODEL_PATH = os.environ["MODEL_PATH"]
TRAIN_FILE = os.environ["TRAIN_FILE"]
VAL_FILE = os.environ.get("VAL_FILE", "")
OUTPUT_DIR = os.environ["OUTPUT_DIR"]
MAX_STEPS = int(os.environ.get("MAX_STEPS", "100"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "2"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "4"))
LORA_RANK = int(os.environ.get("LORA_RANK", "32"))
LR = float(os.environ.get("LR", "5e-5"))
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "512"))
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", "50"))
CKPT_EVERY = int(os.environ.get("CHECKPOINT_EVERY", "20"))

IGNORE = -100


class ChatSFT(Dataset):
    """Chat-template the messages; mask everything before the assistant turn."""

    def __init__(self, path, tok, max_len):
        self.rows, self.tok, self.max_len = [], tok, max_len
        dropped = 0
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            msgs = json.loads(line)["messages"]
            prompt = tok.apply_chat_template(msgs[:-1], tokenize=False,
                                             add_generation_prompt=True)
            full = prompt + msgs[-1]["content"] + (tok.eos_token or "")
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            f_ids = tok(full, add_special_tokens=False)["input_ids"]
            if len(f_ids) > max_len:          # truncating would cut the answer — drop instead
                dropped += 1
                continue
            labels = [IGNORE] * len(p_ids) + f_ids[len(p_ids):]
            self.rows.append({"input_ids": f_ids, "labels": labels})
        print(f"[data] {path}: {len(self.rows)} samples, {dropped} dropped (> {max_len} tok)",
              flush=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    ids, labs, att = [], [], []
    for b in batch:
        k = n - len(b["input_ids"])
        ids.append(b["input_ids"] + [pad_id] * k)
        labs.append(b["labels"] + [IGNORE] * k)
        att.append([1] * len(b["input_ids"]) + [0] * k)
    return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labs),
            "attention_mask": torch.tensor(att)}


def main():
    print(f"[env] torch={torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
    if not torch.cuda.is_available():
        sys.exit("no CUDA device visible inside the container")

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_ds = ChatSFT(TRAIN_FILE, tok, MAX_SEQ_LEN)
    val_ds = ChatSFT(VAL_FILE, tok, MAX_SEQ_LEN) if VAL_FILE and os.path.exists(VAL_FILE) else None
    if len(train_ds) == 0:
        sys.exit("no usable training samples")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_RANK * 2, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.cuda()

    do_eval = val_ds is not None
    targs = TrainingArguments(
        output_dir=OUTPUT_DIR,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_steps=CKPT_EVERY,
        save_total_limit=6,
        bf16=True,
        gradient_checkpointing=True,
        report_to=[],
        dataloader_num_workers=2,
        eval_strategy="steps" if do_eval else "no",
        eval_steps=CKPT_EVERY if do_eval else None,
        per_device_eval_batch_size=BATCH_SIZE,
        remove_unused_columns=False,
        save_safetensors=True,
    )

    trainer = Trainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=lambda b: collate(b, tok.pad_token_id),
    )
    print(f"[train] {MAX_STEPS} steps, effective batch {BATCH_SIZE*GRAD_ACCUM}, "
          f"ckpt every {CKPT_EVERY} -> {OUTPUT_DIR}", flush=True)
    trainer.train()

    # Final adapter in the layout vLLM expects (--lora-modules name=<dir>)
    final = os.path.join(OUTPUT_DIR, "hf_adapter")
    model.save_pretrained(final)
    tok.save_pretrained(final)
    print(f"[done] adapter -> {final}", flush=True)

    # Each intermediate checkpoint also gets an hf_adapter/ so step-20 is servable.
    for d in sorted(os.listdir(OUTPUT_DIR)):
        if d.startswith("checkpoint-"):
            print(f"[ckpt] {os.path.join(OUTPUT_DIR, d)}", flush=True)


if __name__ == "__main__":
    main()
