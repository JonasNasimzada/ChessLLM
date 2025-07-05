from opensloth.opensloth_config import (
    FastModelArgs,
    LoraArgs,
    OpenSlothConfig,
    TrainingArguments,
)
from opensloth.scripts.opensloth_sft_trainer import run_mp_training, setup_envs

# 2 GPUs with packing configuration
GLOBAL_BZ = 32

DEVICES = [0]

BZ = 1  # if sequence packing, then should be 1, larger does not contribute to speed
opensloth_config = OpenSlothConfig(
    data_cache_path="data/cache_Llama-3.2-3B-Instruct",
    dataset_text_field="text",
    devices=DEVICES,
    fast_model_args=FastModelArgs(
        model_name="unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
        max_seq_length=2048,
        load_in_4bit=True,
    ),
    lora_args=LoraArgs(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0,
        bias="none",
        use_rslora=False,
    ),
    sequence_packing=True,
)

training_config = TrainingArguments(
    output_dir="outputs/exps/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
    resume_from_checkpoint="outputs/exps/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
    save_only_model=False,
    max_steps=60,
    per_device_train_batch_size=BZ,
    gradient_accumulation_steps=GLOBAL_BZ // (len(DEVICES) * BZ),
    learning_rate=2e-4,
    logging_steps=1,
    num_train_epochs=3,
    lr_scheduler_type="linear",
    warmup_steps=5,
    save_total_limit=1,
    weight_decay=0.01,
    optim="adamw_8bit",
    seed=3407,
    report_to="wandb",  # or wandb/tensorboard
)

if __name__ == "__main__":
    import os

    print(
        f"Global batch size: {len(DEVICES) * BZ * training_config.gradient_accumulation_steps}"
    )
    print(f"Gradient accumulation steps: {training_config.gradient_accumulation_steps}")

    setup_envs(opensloth_config, training_config)
    run_mp_training(opensloth_config.devices, opensloth_config, training_config)
