# TARVIA LoRA Fine-Tuning

This is a separate weight-level fine-tuning path for an open-source model.

## 1) Install finetune deps

```bash
cd /Users/omarlujanoolazaba/Desktop/Luhano,Inc./tarvia_project
source .venv/bin/activate
pip install -e ".[finetune]"
```

## 2) Build SFT dataset from your gold standards

```bash
python training/prepare_sft_dataset.py \
  --gold-file /Users/omarlujanoolazaba/Desktop/Luhano,Inc./tarvia_project/data/gold_examples.jsonl \
  --out /Users/omarlujanoolazaba/Desktop/Luhano,Inc./tarvia_project/data/sft_train.jsonl
```

## 3) Train LoRA adapter

```bash
python training/train_lora.py \
  --dataset /Users/omarlujanoolazaba/Desktop/Luhano,Inc./tarvia_project/data/sft_train.jsonl \
  --base-model Qwen/Qwen2.5-3B-Instruct \
  --out-dir /Users/omarlujanoolazaba/Desktop/Luhano,Inc./tarvia_project/artifacts/lora_qwen_3b
```

Adapter output:
- `/Users/omarlujanoolazaba/Desktop/Luhano,Inc./tarvia_project/artifacts/lora_qwen_3b/adapter`

## 4) Merge adapter into a full checkpoint (optional)

```bash
python training/merge_lora.py \
  --base-model Qwen/Qwen2.5-3B-Instruct \
  --adapter-dir /Users/omarlujanoolazaba/Desktop/Luhano,Inc./tarvia_project/artifacts/lora_qwen_3b/adapter \
  --out-dir /Users/omarlujanoolazaba/Desktop/Luhano,Inc./tarvia_project/artifacts/merged_qwen_3b_tarvia
```

## Notes

- This fine-tunes model weights. It is independent from the Anthropic API route.
- Keep your current guardrails; do not relax clinical validation when switching models.
