# Modules for fine-tuning
# Hugging Face modules
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig  # Trainer for supervised fine-tuning (SFT)
from datasets import load_dataset  # Lets you load fine-tuning datasets
from unsloth import is_bfloat16_supported  # Checks if the hardware supports bfloat16 precision
from unsloth.chat_templates import get_chat_template
from transformers import DataCollatorForSeq2Seq
from unsloth.chat_templates import train_on_responses_only
from accelerate import PartialState


max_seq_length = 2048  # Define the maximum sequence length a model can handle (i.e. how many tokens can be processed at once)
dtype = None  # Set to default
load_in_4bit = True  # Enables 4 bit quantization — a memory saving optimization
device_string = PartialState().process_index
# Load the DeepSeek R1 model and tokenizer using unsloth — imported using: from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",  # or choose "unsloth/Llama-3.2-1B-Instruct"
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    device_map={'' : device_string},  # Use this to set the device for the model
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj", ],
    lora_alpha=16,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3.1",
)


def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return {"text": texts, }


dataset = load_dataset("json", data_files="data/train_striped_7.json", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True, )

print(dataset[5]["messages"])
print(dataset[5]["text"])

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=3,  # Set this for 1 full training run.
        # max_steps=60,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs_unsloth",
        report_to="wandb",  # Use this for WandB etc
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
    ),
)
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
)

print(tokenizer.decode(trainer.train_dataset[5]["input_ids"]))

space = tokenizer(" ", add_special_tokens=False).input_ids[0]
print(tokenizer.decode([space if x == -100 else x for x in trainer.train_dataset[5]["labels"]]))
trainer_stats = trainer.train()
model.save_pretrained("Llama-3.2-3B-Instruct_unsloth")  # Local saving
tokenizer.save_pretrained("Llama-3.2-3B-Instruct_unsloth")
model.push_to_hub("Llama-3.2-3B-Instruct")  # Online saving
tokenizer.push_to_hub("Llama-3.2-3B-Instruct")  # Online saving
