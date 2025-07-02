# Modules for fine-tuning
# Hugging Face modules
from datasets import load_dataset  # Lets you load fine-tuning datasets
from trl import SFTTrainer, SFTConfig  # Trainer for supervised fine-tuning (SFT)
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported  # Checks if the hardware supports bfloat16 precision


max_seq_length = 2048  # Define the maximum sequence length a model can handle (i.e. how many tokens can be processed at once)
dtype = None  # Set to default
load_in_4bit = True  # Enables 4 bit quantization — a memory saving optimization

# Load the DeepSeek R1 model and tokenizer using unsloth — imported using: from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/DeepSeek-R1-Distill-Llama-8B",  # Load the pre-trained DeepSeek R1 model (8B parameter version)
    max_seq_length=max_seq_length,  # Ensure the model can process up to 2048 tokens at once
    dtype=dtype,  # Use the default data type (e.g., FP16 or BF16 depending on hardware support)
    load_in_4bit=load_in_4bit,  # Load the model in 4-bit quantization to save memory
)
dataset = load_dataset("data")
EOS_TOKEN = tokenizer.eos_token  # Define EOS_TOKEN which the model when to stop generating text during training

model_lora = FastLanguageModel.get_peft_model(
    model,
    r=16,
    # LoRA rank: Determines the size of the trainable adapters (higher = more parameters, lower = more efficiency)
    target_modules=[  # List of transformer layers where LoRA adapters will be applied
        "q_proj",  # Query projection in the self-attention mechanism
        "k_proj",  # Key projection in the self-attention mechanism
        "v_proj",  # Value projection in the self-attention mechanism
        "o_proj",  # Output projection from the attention layer
        "gate_proj",  # Used in feed-forward layers (MLP)
        "up_proj",  # Part of the transformer’s feed-forward network (FFN)
        "down_proj",  # Another part of the transformer’s FFN
    ],
    lora_alpha=16,  # Scaling factor for LoRA updates (higher values allow more influence from LoRA layers)
    lora_dropout=0,  # Dropout rate for LoRA layers (0 means no dropout, full retention of information)
    bias="none",  # Specifies whether LoRA layers should learn bias terms (setting to "none" saves memory)
    use_gradient_checkpointing="unsloth",
    # Saves memory by recomputing activations instead of storing them (recommended for long-context fine-tuning)
    random_state=3407,  # Sets a seed for reproducibility, ensuring the same fine-tuning behavior across runs
    use_rslora=False,  # Whether to use Rank-Stabilized LoRA (disabled here, meaning fixed-rank LoRA is used)
    loftq_config=None,  # Low-bit Fine-Tuning Quantization (LoFTQ) is disabled in this configuration
)

trainer = SFTTrainer(
    model=model_lora,  # The model to be fine-tuned
    processing_class=tokenizer,  # Tokenizer to process text inputs
    train_dataset=dataset['train'],  # Dataset used for training

    # Define training arguments
    args=SFTConfig(
        per_device_train_batch_size=2,  # Number of examples processed per device (GPU) at a time
        gradient_accumulation_steps=4,  # Accumulate gradients over 4 steps before updating weights
        num_train_epochs=1,  # Full fine-tuning run
        warmup_steps=5,  # Gradually increases learning rate for the first 5 steps
        max_steps=60,  # Limits training to 60 steps (useful for debugging; increase for full fine-tuning)
        learning_rate=2e-4,  # Learning rate for weight updates (tuned for LoRA fine-tuning)
        fp16=not is_bfloat16_supported(),  # Use FP16 (if BF16 is not supported) to speed up training
        bf16=is_bfloat16_supported(),  # Use BF16 if supported (better numerical stability on newer GPUs)
        logging_steps=10,  # Logs training progress every 10 steps
        optim="adamw_8bit",  # Uses memory-efficient AdamW optimizer in 8-bit mode
        weight_decay=0.01,  # Regularization to prevent overfitting
        lr_scheduler_type="linear",  # Uses a linear learning rate schedule
        seed=3407,  # Sets a fixed seed for reproducibility
        output_dir="outputs",  # Directory where fine-tuned model checkpoints will be saved
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        use_liger_kernel=True,
        report_to="wandb"
    ),
)

trainer_stats = trainer.train()
