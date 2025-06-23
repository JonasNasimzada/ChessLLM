from datasets import load_dataset, Dataset
from huggingface_hub import login
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import setup_chat_format, SFTTrainer
from peft import LoraConfig
from transformers import TrainingArguments
import pandas as pd

system_message = """You are the best chest engine that plays chess games. As an input u get the current chess 
position postions of the past moves in FEN notation. You will generate the best chess move in UCI format. Here are 
the FEN notation of the past moves: {moves}"""

user_message = """Current chess position in FEN notation: {fen} - what is the next best move in UCI 
format?"""


def instruction_format(sample):
    return {
        "messages": [
            {"role": "system", "content": system_message.format(moves=sample["context"])},
            {"role": "user", "content": user_message.format(fen=sample["fen"])},
            {"role": "assistant", "content": sample["move"]}
        ]
    }


def create_dataset():
    dataset = load_dataset("../data/", split="train")
    ds = dataset.shuffle()
    df = pd.DataFrame(ds['train'])
    df = df.sort_values(["game_index", "ply_index"])

    def add_context(group):
        fens = group["fen"].tolist()
        contexts = []
        for i in range(len(fens)):
            start = max(0, i - 27)
            # join the previous fens (i.e. positions before the current ply)
            contexts.append(" ".join(fens[start:i]))
        group["context"] = contexts
        return group

    df = df.groupby("game_index", group_keys=False).apply(add_context)
    ds = Dataset.from_pandas(df)


    # Convert dataset to OAI messages
    dataset = dataset.map(instruction_format, remove_columns=dataset.features, batched=False)
    # split dataset into 10,000 training samples and 2,500 test samples
    dataset = dataset.train_test_split(train_size=0.8, test_size=0.2)

    print(dataset["train"][345]["messages"])

    # save datasets to disk
    dataset["train"].to_json("train_dataset.json", orient="records")
    dataset["test"].to_json("test_dataset.json", orient="records")
    return dataset


if __name__ == "__main__":
    login(
        token="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",
        add_to_git_credential=True
    )

    model_id = "openlm-research/open_llama_3b_v2"

    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True
    )
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained("model_pretrained_chess_llm", safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = 'right'  # to prevent warnings

    # # set chat template to OAI chatML, remove if you start from a fine-tuned model
    model, tokenizer = setup_chat_format(model, tokenizer)

    peft_config = LoraConfig(
        lora_alpha=128,
        lora_dropout=0.05,
        r=256,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    args = TrainingArguments(
        output_dir="pretrained_chess_llm",  # directory to save and repository id
        num_train_epochs=3,  # number of training epochs
        per_device_train_batch_size=3,  # batch size per device during training
        gradient_accumulation_steps=2,  # number of steps before performing a backward/update pass
        gradient_checkpointing=True,  # use gradient checkpointing to save memory
        optim="adamw_torch_fused",  # use fused adamw optimizer
        logging_steps=10,  # log every 10 steps
        save_strategy="epoch",  # save checkpoint every epoch
        learning_rate=2e-4,  # learning rate, based on QLoRA paper
        bf16=True,  # use bfloat16 precision
        tf32=True,  # use tf32 precision
        max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,  # warmup ratio based on QLoRA paper
        lr_scheduler_type="constant",  # use constant learning rate scheduler
        push_to_hub=True,  # push model to hub
        report_to="wandb",  # report metrics to tensorboard
    )

    max_seq_length = 2048  # max sequence length for model and packing of the dataset

    dataset = create_dataset()  # load or create dataset
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        dataset_kwargs={
            "add_special_tokens": False,  # We template with special tokens
            "append_concat_token": False,  # No need to add additional separator token
        }
    )

    trainer.train()

    # save model
    trainer.save_model()
