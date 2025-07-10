# ChessLLM

# Required packages

`conda create -n chessllm python=3.9 -y`

`conda activate chessllm`

`pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124`

`pip install python-chess stockfish pandas wandb`

## Additional packages for pretraining & reinforcement learning the LLM
`pip install transformers datasets accelerate evaluate bitsandbytes sentencepiece protobuf tiktoken trl peft accelerate importlib-metadata liger-kernel trl[vllm] wandb`

### Optional packages for training the LLM with FlashAttention for faster training
`pip install ninja packaging`

`MAX_JOBS=8 pip install flash-attn --no-build-isolation`