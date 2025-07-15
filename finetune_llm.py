"""
This script fine-tunes a pre-trained language model using a custom dataset.
It loads the model and tokenizer via `FastLanguageModel`, applies PEFT (LoRA),
formats the dataset with a chat template, and trains with `SFTTrainer`.
Finally, it saves and pushes the fine-tuned model and tokenizer to the hub.
"""

import argparse
from accelerate import PartialState
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only

if __name__ == "__main__":
    # Parse command-line arguments for model, output paths, and dataset
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="unsloth/Llama-3.2-3B-Instruct",
        required=False,
        help="Name of the pre-trained model to fine-tune."
    )
    parser.add_argument(
        "--finetune_model",
        type=str,
        default="finetuned_model",
        required=False,
        help="Name for the new fine-tuned model to save and push."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/finetuned",
        required=False,
        help="Directory to store fine-tuning outputs."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        default="finetune_dataset.json",
        help="Path to JSON file containing prompts and completions."
    )
    args = parser.parse_args()

    # Training configuration
    max_seq_length = 2048  # Maximum sequence length
    dtype = None  # Data type (e.g., float16 or float32)
    load_in_4bit = True  # Load model in 4-bit precision
    device_string = PartialState().process_index  # Device mapping index

    # Load pre-trained model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        device_map={"": device_string}
    )

    # Apply PEFT (LoRA) to model for parameter-efficient fine-tuning
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # Wrap tokenizer with chat template for prompt formatting
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")


    def formatting_prompts_func(examples):
        """
        Map function to format dataset examples by applying chat template.

        Args:
            examples (dict): Batch of examples with key "messages".

        Returns:
            dict: Batch with new key "text" containing formatted strings.
        """
        convos = examples["messages"]
        texts = [
            tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False
            )
            for convo in convos
        ]
        return {"text": texts}


    # Load dataset and apply formatting function
    dataset = load_dataset("json", data_files=args.dataset, split="train")
    dataset = dataset.map(formatting_prompts_func, batched=True)

    # Display sample for verification
    print(dataset[5]["messages"])
    print(dataset[5]["text"])

    # Initialize SFT trainer with training arguments
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=3,
            learning_rate=2e-4,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=args.output,
            report_to="wandb",
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            dataset_num_proc=2,
            packing=False,
            ddp_find_unused_parameters=False,
        ),
    )

    # Configure trainer to train on responses only
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    # Show tokenized inputs and labels sample for debugging
    print(tokenizer.decode(trainer.train_dataset[5]["input_ids"]))
    space = tokenizer(" ", add_special_tokens=False).input_ids[0]
    print(
        tokenizer.decode(
            [space if x == -100 else x for x in trainer.train_dataset[5]["labels"]]
        )
    )

    # Train the model and capture statistics
    trainer_stats = trainer.train()

    # Save and push fine-tuned model and tokenizer to hub
    model.save_pretrained(args.finetune_model)
    tokenizer.save_pretrained(args.finetune_model)
    model.push_to_hub(args.finetune_model)
    tokenizer.push_to_hub(args.finetune_model)
