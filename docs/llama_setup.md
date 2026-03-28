# Llama 3 8B Instruct - Local Setup Guide

Kimodo uses the LLM2Vec text encoder based on Meta-Llama-3-8B-Instruct.
This document explains how to obtain and set up the model weights locally.

## Model Chain

```
meta-llama/Meta-Llama-3-8B-Instruct          (base LLM, 15GB)
    |
    v  LoRA adapter (MNTP fine-tune)
McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp       (161MB adapter)
    |
    v  LoRA adapter (supervised)
McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised  (161MB adapter)
```

All three are needed. The McGill-NLP models are public; the base Llama model is gated.

## Two Routes to Obtain Llama Weights

### Route A: HuggingFace (recommended if approved)

1. Request access at https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
2. Wait for approval (can take hours to days)
3. Model is downloaded automatically by `transformers` on first use

### Route B: Meta Direct Download (if HF approval is pending)

1. Request access at https://www.llama.com/llama-downloads/
2. Receive a signed download URL via email (valid for 24 hours)
3. Download using the original Llama 3 download script:

```bash
# The llama-stack CLI uses WRONG file paths for Llama 3.
# Use curl with the correct paths from the original download.sh:
PRESIGNED_URL='<your-url-from-email>'
TARGET="$HOME/.llama/checkpoints/Meta-Llama-3-8B-Instruct"
mkdir -p "$TARGET"

curl -o "$TARGET/params.json"          "${PRESIGNED_URL/'*'/8b_instruction_tuned/params.json}"
curl -o "$TARGET/tokenizer.model"      "${PRESIGNED_URL/'*'/8b_instruction_tuned/tokenizer.model}"
curl -o "$TARGET/checklist.chk"        "${PRESIGNED_URL/'*'/8b_instruction_tuned/checklist.chk}"
curl -o "$TARGET/consolidated.00.pth"  "${PRESIGNED_URL/'*'/8b_instruction_tuned/consolidated.00.pth}"
```

4. Convert from Meta format to HuggingFace format:

```bash
python convert_llama_to_hf.py
```

This produces `~/.llama/hf/Meta-Llama-3-8B-Instruct/` with `model.safetensors`, `config.json`, and tokenizer files.

5. Download McGill-NLP adapter models:

```python
from huggingface_hub import snapshot_download
snapshot_download("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
                  local_dir="~/.llama/hf/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp")
snapshot_download("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
                  local_dir="~/.llama/hf/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised")
```

6. Update `adapter_config.json` in the McGill MNTP model to point to the local base model:

```json
"base_model_name_or_path": "C:/Users/<you>/.llama/hf/Meta-Llama-3-8B-Instruct"
```

### Important: llama-stack CLI Does NOT Work for Llama 3

The `llama model download` command from `llama-stack` constructs S3 paths like
`Llama-3-8B-Instruct/tokenizer.model`, but the actual Llama 3 server uses
`8b_instruction_tuned/tokenizer.model`. This causes 403 errors.
The path mapping comes from the original `download.sh` in `meta-llama/llama3`:

| CLI model ID | Actual S3 path |
|---|---|
| 8B | `8b_pre_trained/` |
| 8B-instruct | `8b_instruction_tuned/` |
| 70B | `70b_pre_trained/` |
| 70B-instruct | `70b_instruction_tuned/` |

## Conversion Verification

The converted weights (Route B) were compared against the official HuggingFace
weights (Route A) tensor by tensor:

```
Key comparison:
  Common:         291
  Only converted: 0
  Only official:  0

Comparing 291 common tensors...

RESULT: ALL TENSORS IDENTICAL (bit-for-bit match)
```

The conversion is lossless. Key details of the conversion:
- Q/K attention weights require permutation (dimension reordering)
  between Meta and HuggingFace formats
- The `permute()` function handles the interleaved head layout
- All 291 tensors match exactly with zero difference

## Usage

```bash
# With local models (Route B)
set KIMODO_QUANTIZE=4bit
set TEXT_ENCODER_MODE=local
set TEXT_ENCODERS_DIR=C:\Users\<you>\.llama\hf
kimodo_gen "A person walks forward." --output motion

# With HuggingFace (Route A, after approval)
set KIMODO_QUANTIZE=4bit
kimodo_gen "A person walks forward." --output motion
```

## VRAM Requirements

| Mode | VRAM | Text Prompts |
|---|---|---|
| `TEXT_ENCODER_MODE=dummy` | ~1.1 GB | No |
| `KIMODO_QUANTIZE=4bit` | ~5.6 GB | Yes |
| `KIMODO_QUANTIZE=8bit` | ~9 GB | Yes |
| Full precision | ~15.7 GB | Yes |
