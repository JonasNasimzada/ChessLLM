# ChessLLM

## Install Environment

`conda create -n chessllm python=3.10 -y`

`conda activate chessllm`

### Required packages

`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`

`pip install python-chess stockfish pandas wandb`

#### Additional packages for finetuning & reinforcement learning the LLM

`pip install unsloth transformers datasets accelerate evaluate sentencepiece protobuf tiktoken trl peft importlib-metadata vllm`

#### Optional packages for training the LLM with FlashAttention for faster training

`pip install ninja packaging`

`MAX_JOBS=8 pip install flash-attn --no-build-isolation`

## Overview
```
.
├── pretrain_transformer.py     # Supervised pre-training for transformer policy network
├── ppo_transformer.py          # RL fine-tuning of transformer policy (episodic training; W&B logging)
├── train_grpo.py               # GRPO prompt generation and dataset creation for LLM fine-tuning
├── inference_transformer.py    # Inference-time move selection using trained transformer policy
└── inference_llm.py            # Inference-time move generation using fine-tuned language model
```


## Download the Stockfish binary
Download the Stockfish binary for your system from the [official repository](https://stockfishchess.org/download/).:

After downloading, unpack it with:
```bash
tar -xvf stockfish-ubuntu-x86-64-avx2.tar.gz
```

