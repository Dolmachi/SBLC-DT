# SBLC-DT

**Self-Building Local Conversational Digital Twin**

SBLC-DT builds a local conversational digital twin from real dialogue recordings, a short text profile, a clean voice reference, and one avatar image.

## 🌟 Features

- **Profile builder** — one command prepares a complete digital-twin profile.
- **ASR + diarization** — WhisperX, faster-whisper, and pyannote-based speaker processing.
- **Target speaker reference** — `reference.*` is used to find the target speaker in real dialogs.
- **Dialogue RAG** — extracted dialog pairs are indexed in ChromaDB.
- **Voice adaptation** — clean target speech segments are turned into a VoxCPM2 dataset and LoRA adapter.
- **Avatar output** — prepared artifacts for SoulX-FlashHead.
- **Local runtime** — microphone input, LLM response, TTS output, memory, and avatar playback run locally.

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Input Data Format](#input-data-format)
- [Pipeline](#pipeline)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
- [Project Structure](#project-structure)
- [Supported Languages](#supported-languages)
- [Notes](#notes)
- [Acknowledgments](#acknowledgments)

## 🛠️ Installation

Target setup used for the project:

- Ubuntu / Linux
- Python 3.11
- NVIDIA GPU with CUDA
- `ffmpeg`
- Hugging Face token with access to the required diarization models

1. Clone the repository:

```bash
git clone https://github.com/Dolmachi/SBLC-DT.git
cd SBLC-DT
```

2. Install system dependencies:

```bash
sudo apt update
sudo apt install -y ffmpeg portaudio19-dev
```

3. Create and activate a virtual environment:

```bash
uv venv --python 3.11
source .venv/bin/activate
```

4. Install PyTorch:

```bash
uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 torchcodec==0.7.0 \
    --index-url https://download.pytorch.org/whl/cu128
```

5. Install project dependencies:

```bash
uv pip install -r requirements.txt
uv pip install transformers huggingface_hub
```

6. Install `flash-attn` for the CUDA setup used by the project:

```bash
uv pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
```

If the wheel does not match your environment, build it from source:

```bash
uv pip install flash-attn==2.7.3 --no-build-isolation
```

7. Save the Hugging Face token:

```bash
python scripts/setup.py --hf-token hf_xxx
```

8. Download model weights:

```bash
bash scripts/download_models.sh
```

## 🚀 Quick Start

1. Prepare the input folder:

```text
person_data/
├── dialogs/
│   ├── interview_01.mp4
│   ├── interview_02.wav
│   └── call_03.mp3
├── profile.txt
├── reference.wav
└── avatar.jpg
```

2. Build a profile:

```bash
python scripts/train.py \
    --name "James Smith" \
    --lang en \
    --data /path/to/person_data
```

3. Run the local clone:

```bash
python scripts/clone.py
```

Or open a profile directly:

```bash
python scripts/clone.py \
    --profile profiles/james_smith
```

## 📦 Input Data Format

### `dialogs/`

Audio or video recordings with the target person.

Supported audio formats:

```text
.wav, .flac, .mp3, .m4a, .aac, .ogg, .opus, .wma
```

Supported video formats:

```text
.mp4, .mkv, .mov, .avi, .webm, .m4v, .mpg, .mpeg, .wmv, .3gp
```

During preprocessing, dialogs are converted to 16 kHz mono `.wav` files. Broken, silent, duplicate, or too short files are filtered out.

### `profile.txt`

A short text description of the target person.

Example:

```text
James is a university professor. He speaks calmly, explains things step by step, and prefers precise wording.
```

Use the same language as the selected `--lang` when possible.

### `reference.*`

A clean voice reference of the target speaker.

Requirements:

- exactly one `reference.*` file in the input folder;
- audio only;
- 3–30 seconds long;
- no background speakers if possible.

This file is used to identify the target speaker in the dialogs and as the TTS reference sample.

### `avatar.*`

One source image for the avatar.

Supported names:

```text
avatar.jpg
avatar.jpeg
avatar.png
avatar.webp
avatar.bmp
```

Only one avatar file should be present.

## 🧩 Pipeline

### Training

```text
input data
   ↓
init_profile
   ↓
preprocess_source
   ↓
asr_ingest
   ↓
asr_postprocess
   ↓
dialog_pairs
   ↓
rag_build
   ↓
tts_dataset_build
   ↓
tts_prepare
   ↓
avatar_prepare
   ↓
profiles/<slug>/
```

### Inference

```text
microphone
   ↓
faster-whisper ASR
   ↓
Qwen3.5 + context-aware RAG + selected dialog memory
   ↓
VoxCPM2 + LoRA adapter
   ↓
SoulX-FlashHead
   ↓
local audio/video output
```

## 📖 Usage

### Training

```bash
python scripts/train.py \
    --name "James Smith" \
    --lang en \
    --data /path/to/person_data
```

### Inference

Run with interactive profile and dialog selection:

```bash
python scripts/clone.py
```

Run a specific profile by path:

```bash
python scripts/clone.py \
    --profile profiles/james_smith
```

Run a specific profile by name:

```bash
python scripts/clone.py \
    --name "James Smith"
```

Open an existing dialog memory:

```bash
python scripts/clone.py \
    --profile profiles/james_smith \
    --dialog-title dialog_0
```

Optional model arguments:

```bash
python scripts/clone.py \
    --profile profiles/james_smith \
    --llm-model Qwen/Qwen3.5-4B \
    --asr-model large-v3-turbo \
    --tts-model openbmb/VoxCPM2
```

### Profile and Dialog Management

Delete a dialog history:

```bash
python scripts/clone.py \
    --profile profiles/james_smith \
    --dialog-title dialog_0 \
    --delete-dialog
```

Delete a profile:

```bash
python scripts/clone.py \
    --profile profiles/james_smith \
    --delete-profile
```

## 📁 Project Structure

```text
.
├── scripts/
│   ├── train.py              # Build a digital-twin profile
│   ├── clone.py              # Run inference and manage profiles/dialogs
│   ├── setup.py              # Save Hugging Face token
│   └── download_models.sh    # Download external model weights
├── src/
│   ├── asr/                  # Training and runtime ASR backends
│   ├── avatar/               # SoulX-FlashHead integration
│   ├── inference/            # Local runtime session, audio I/O, window output
│   ├── llm/                  # Qwen runtime, LangGraph memory, RAG calls
│   ├── rag/                  # RAG formatting and runtime embedding modules
│   ├── training/             # Profile-building pipeline
│   │   └── stages/           # Individual training stages
│   ├── tts/                  # VoxCPM2 backend
│   └── utils/                # Configs, paths, logging, helpers
├── models/                   # Downloaded model weights, not committed
├── profiles/                 # Generated profiles
└── requirements.txt
```

## 🌍 Supported Languages

The language is selected with `--lang`.

Supported language codes:

```text
en, ru, es, de, fr, it, el, pl, pt, fi, sv, nl, da, no, he, tr, ar, hi, zh, ja, ko, tl, vi
```

## ⚠️ Notes

- This is a research prototype, not a production system.
- A CUDA-capable NVIDIA GPU is expected for the full pipeline.
- Output quality depends on the amount and quality of dialog recordings, speaker diarization, and the reference voice sample.
- Use the project only with data you are allowed to process and with consent from the person being cloned.
- Running training again with the same `--name` recreates the corresponding `profiles/<slug>/` directory from scratch.

## 🙏 Acknowledgments

This project builds on open-source tools and models from the ASR, LLM, RAG, voice synthesis, and avatar generation communities:

- WhisperX
- faster-whisper
- pyannote
- Qwen
- VoxCPM2
- SoulX-FlashHead
- LangChain
- LangGraph
- ChromaDB
- Hugging Face Transformers