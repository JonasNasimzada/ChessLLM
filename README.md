# ChessLLM

# Install Environment
## Required packages

`conda create -n chessllm python=3.10 -y`

`conda activate chessllm`

`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`

`pip install python-chess stockfish pandas wandb`

### Additional packages for finetuning & reinforcement learning the LLM

`pip install unsloth transformers datasets accelerate evaluate sentencepiece protobuf tiktoken trl peft importlib-metadata vllm`

### Optional packages for training the LLM with FlashAttention for faster training

`pip install ninja packaging`

`MAX_JOBS=8 pip install flash-attn --no-build-isolation`