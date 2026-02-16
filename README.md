# Full Fine-Tuning LLM on RTX 3090

## Why Full Fine-Tuning (vs LoRA/QLoRA)?

| Aspect                 | Full Fine-Tuning           | QLoRA (what your notebooks do) |
|------------------------|----------------------------|--------------------------------|
| Parameters updated     | **ALL**                    | ~0.5-2% (adapter only)         |
| VRAM needed (1B model) | ~16-20 GB                  | ~6-8 GB                        |
| Quality ceiling        | Higher (all weights adapt) | Good but constrained           |
| Output size            | Full model (~2-4 GB)       | Small adapter (~100 MB)        |
| Use case               | Max quality, domain shift  | Quick adaptation, low resource |

## Hardware Constraint: RTX 3090 (24 GB VRAM)

**Rule of thumb:** Full fine-tuning needs ~16 bytes/parameter (fp16 weights + gradients + AdamW states).

- **1B params × 16 bytes = 16 GB** → fits with gradient checkpointing + bf16
- **2B+ params** → does NOT fit on a single 3090

**Recommended model: `meta-llama/Llama-3.2-1B-Instruct`** (1.24B params)

### Memory-Saving Tricks Used
1. **Gradient Checkpointing** — recomputes activations instead of storing them (saves ~40% VRAM)
2. **BF16 Mixed Precision** — half-precision training (saves ~50% vs fp32)
3. **Small batch size + gradient accumulation** — batch_size=1, accumulate over 8 steps = effective batch of 8

## Project Structure

```
full-finetune-llm/
├── README.md                  ← You are here
├── requirements.txt           ← Python dependencies
├── configs/
│   └── training_config.yaml   ← All hyperparameters in one place
├── scripts/
│   ├── train.py               ← Main training script
│   ├── prepare_data.py        ← Dataset loading & formatting
│   ├── inference.py           ← Test your fine-tuned model
│   └── check_vram.py          ← Pre-flight VRAM estimation
├── data/                      ← Your datasets go here (gitignored)
└── outputs/                   ← Saved models land here
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Check if your model fits in VRAM
python scripts/check_vram.py

# 3. Train
python scripts/train.py

# 4. Test inference
python scripts/inference.py --prompt "What is machine learning?"
```

## Key Difference from Your QLoRA Notebooks

Your existing notebooks do this:
```
Base Model (frozen, 4-bit) → LoRA Adapter (tiny, trainable) → Save adapter only
```

This project does this:
```
Base Model (ALL weights trainable, bf16) → Full training → Save entire model
```

No `peft`, no `LoraConfig`, no `BitsAndBytesConfig`. Every single weight gets updated.
