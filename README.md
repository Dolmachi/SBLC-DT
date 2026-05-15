# 🧬 Self-Building Local Conversational Digital Twin

A local, privacy-preserving pipeline for automatically building an interactive conversational digital twin from dialogue recordings, a textual profile, and a reference image.

The system takes real audio/video conversations, a short profile description, and a single avatar image. It then builds a local profile that can answer in the target person's conversational style, combine general language-model knowledge with personal facts retrieved from the target's past conversations, synthesize speech in a similar voice, and animate a talking-head avatar.

The project is focused on privacy-preserving digital personality preservation: all data, models, intermediate artifacts, RAG databases, and runtime memory are stored and processed locally.

## ✨ Features

- **Self-building digital twin pipeline** from dialogues, profile text, and a reference image
- **Modular architecture** with separate ASR, embeddings, LLM/RAG, TTS, avatar, training, and inference components
- **Speech recognition and diarization** with WhisperX, faster-whisper, and pyannote
- **Automatic dialogue dataset extraction** from real conversations
- **Personal knowledge retrieval** using ChromaDB-based RAG
- **Multilingual profile support** with language-specific embeddings, labels, and prompts
- **Conversational LLM runtime** powered by Qwen3.5
- **Long-term dialogue memory** with LangGraph and SQLite
- **Voice cloning** with VoxCPM2
- **Talking-head avatar generation** with SoulX-FlashHead
- **Interactive local interface** with microphone input, text streaming, speech output, and avatar playback

## 🧩 Pipeline Overview

### 🏗️ Training

```text
input data
  ├── dialogs/          audio/video conversations
  ├── profile.txt       short description of the target person
  └── avatar.jpg        avatar source image
        ↓
preprocess dialogs
        ↓
WhisperX ASR + diarization
        ↓
ASR postprocessing
        ↓
dialog pair extraction
        ↓
RAG database build
        ↓
target voice segment selection
        ↓
TTS reference preparation
        ↓
FlashHead avatar artifact preparation
        ↓
profiles/<person_slug>/
```

### 💬 Inference

```text
microphone
   ↓
faster-whisper ASR
   ↓
Qwen3.5 + RAG + dialogue memory
   ↓
VoxCPM2 TTS
   ↓
SoulX-FlashHead avatar animation
   ↓
local OpenCV window + audio playback
```

## 🛠️ Installation

The project is currently tested on Ubuntu with Python 3.11 and RTX 4090.

### 1. Clone the repository

```bash
git clone https://github.com/Dolmachi/Virtual_Clone.git
cd Virtual_Clone
```

### 2. Install system dependencies

On Ubuntu:

```bash
sudo apt update
sudo apt install -y ffmpeg portaudio19-dev
```

`ffmpeg` is used for audio/video preprocessing.  
`portaudio19-dev` is required by `sounddevice` for microphone input and audio playback.

### 3. Create a virtual environment

```bash
uv venv --python 3.11
source .venv/bin/activate
```

### 4. Install PyTorch

```bash
uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 torchcodec==0.7.0 --index-url https://download.pytorch.org/whl/cu128
```

### 5. Install project dependencies

```bash
uv pip install -r requirements.txt
```

### 6. Install Transformers separately

```bash
uv pip install "transformers==5.3.0"
```

Transformers is installed separately because the project currently uses a newer Transformers version together with WhisperX. Installing both in a single command may lead to dependency conflicts.

### 7. Install flash-attn

```bash
uv pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
```

This wheel is specific to Python 3.11, CUDA 12, and PyTorch 2.8.

To build from source:

```bash
uv pip install flash-attn==2.7.3 --no-build-isolation
```

### 8. Configure Hugging Face token

WhisperX diarization uses pyannote models from Hugging Face. You need a Hugging Face access token with access to the required pyannote models.

Save the token with:

```bash
python scripts/setup.py --hf-token hf_xxx
```

### 9. Download model weights

```bash
bash scripts/download_models.sh
```

## 📦 Input Data Format

Prepare a folder with the following structure:

```text
my_person_data/
├── dialogs/
│   ├── interview_01.mp4
│   ├── interview_02.wav
│   └── conversation_03.mp3
├── profile.txt
└── avatar.jpg
```

### `dialogs/`

The `dialogs` folder may contain audio or video files.

Supported audio formats:

```text
.wav, .flac, .mp3, .m4a, .aac, .ogg, .opus, .wma
```

Supported video formats:

```text
.mp4, .mkv, .mov, .avi, .webm, .m4v, .mpg, .mpeg, .wmv, .3gp
```

Current target-speaker detection is heuristic: the pipeline assumes that the target person is usually the **SECOND** speaker who appears in the dialogue. Interview-style recordings work best.

### `profile.txt`

A short natural-language description of the target person.

Example:

```text
James is a university professor and researcher. He usually speaks calmly and thoughtfully, often gives detailed explanations, and prefers precise wording.
```

Use the same language as the target profile when possible.

### `avatar.*`

A single image used as the avatar source.

Supported names:

```text
avatar.jpg
avatar.jpeg
avatar.png
avatar.webp
avatar.bmp
```

Only one avatar file should be present in the input folder.

## 🚀 Training

Run:

```bash
python scripts/train.py \
    --name "James Smith" \
    --lang en \
    --data /path/to/my_person_data
```

The output profile will be created in:

```text
profiles/<slug>/
```

Example:

```text
profiles/james_smith/
├── source/
├── train_data/
├── artifacts/
├── memory/
├── logs/
└── config.json
```

## 🖥️ Inference

Run the local clone:

```bash
python scripts/clone.py \
    --profile profiles/james_smith
```

Optional arguments:

```bash
python scripts/clone.py \
    --profile profiles/james_smith \
    --llm-model Qwen/Qwen3.5-4B \
    --asr-model large-v3-turbo \
    --tts-model openbmb/VoxCPM2
```

## 🌍 Multilingual Support

The project currently supports the following languages:

| Code | Language |
| --- | --- |
| `en` | English |
| `ru` | Russian |
| `es` | Spanish |
| `de` | German |
| `fr` | French |
| `it` | Italian |
| `el` | Greek |
| `pl` | Polish |
| `pt` | Portuguese |
| `fi` | Finnish |
| `sv` | Swedish |
| `nl` | Dutch |
| `da` | Danish |
| `no` | Norwegian |
| `he` | Hebrew |
| `tr` | Turkish |
| `ar` | Arabic |
| `hi` | Hindi |
| `zh` | Chinese |
| `ja` | Japanese |
| `ko` | Korean |
| `tl` | Tagalog / Filipino |
| `vi` | Vietnamese |

## 📁 Project Structure

```text
Virtual_Clone/
├── scripts/
│   ├── train.py              # Build a virtual clone profile
│   ├── clone.py              # Run local inference
│   ├── setup.py              # Save HF token into .env
│   └── download_models.sh    # Download model weights
├── src/
│   ├── asr/                  # ASR interfaces and WhisperX/faster-whisper backends
│   ├── avatar/               # FlashHead avatar runtime and vendored code
│   ├── embeddings/           # Multilingual embedding registry
│   ├── inference/            # Runtime pipeline, session, audio I/O, local window
│   ├── llm/                  # Qwen3.5 + RAG logic
│   ├── training/             # Training pipeline and stages
│   ├── tts/                  # VoxCPM2 TTS backend
│   └── utils/                # Config, paths, logging, filesystem helpers
├── models/                   # Downloaded model weights, not committed
├── profiles/                 # Generated clone profiles, not committed
└── requirements.txt
```

## ⚠️ Limitations

- The current system is a research prototype, not a production product.
- The project currently requires an NVIDIA GPU with CUDA support and at least 24 GB of VRAM.
- Output quality depends strongly on the quality, duration, and structure of the input recordings.
- The project should only be used with data that you have the right to process.

## 🛡️ Ethics

This project is intended for research in local, privacy-preserving virtual doubles and digital personality preservation.

Use it only with explicit consent from the person being cloned or with data that you are legally and ethically allowed to process.

Do not use this project for impersonation, fraud, harassment, or deceptive synthetic media.

## 🙏 Acknowledgments

This project builds on open-source tools and models from the ASR, speaker diarization, LLM, RAG, voice synthesis, and avatar generation communities, including:

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
