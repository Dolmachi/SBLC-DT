#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/models}"

FLASHHEAD_MODELS_DIR="$MODEL_DIR/flashhead"
FLASHHEAD_CKPT_DIR="$FLASHHEAD_MODELS_DIR/SoulX-FlashHead-1_3B"
WAV2VEC_DIR="$FLASHHEAD_MODELS_DIR/wav2vec2-base-960h"

mkdir -p "$FLASHHEAD_CKPT_DIR" "$WAV2VEC_DIR"

if ! command -v hf >/dev/null 2>&1; then
  echo "Команда 'hf' не найдена. Установи пакет huggingface_hub:" >&2
  echo "  pip install huggingface_hub" >&2
  exit 1
fi

HF_CMD="hf"

echo "==> SoulX-FlashHead-1_3B lite/pro weights"
$HF_CMD download Soul-AILab/SoulX-FlashHead-1_3B \
  --local-dir "$FLASHHEAD_CKPT_DIR" \
  --include "config.json" \
  --include "model_index.json" \
  --include "Model_Lite/*" \
  --include "Model_Pro/*" \
  --include "VAE_LTX/*" \
  --include "VAE_Wan/*"

echo "==> wav2vec2-base-960h"
$HF_CMD download facebook/wav2vec2-base-960h \
  --local-dir "$WAV2VEC_DIR" \
  --include "config.json" \
  --include "preprocessor_config.json" \
  --include "feature_extractor_config.json" \
  --include "tokenizer_config.json" \
  --include "special_tokens_map.json" \
  --include "vocab.json" \
  --include "model.safetensors" \
  --include "pytorch_model.bin"

echo
echo "Готово."
echo "FlashHead weights: $FLASHHEAD_CKPT_DIR"
echo "Wav2Vec2:          $WAV2VEC_DIR"
