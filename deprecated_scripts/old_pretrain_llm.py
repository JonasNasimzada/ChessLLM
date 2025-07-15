####################################################################################
# THIS SCRIPT IS DEPRECATED AND NO LONGER USED. IT IS LEFT HERE FOR REFERENCE ONLY.#
####################################################################################

import torch
from accelerate import PartialState
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import setup_chat_format, SFTTrainer, SFTConfig

if __name__ == "__main__":
    model_id = "openlm-research/open_llama_3b_v2"

    base_model = AutoModelForCausalLM.from_pretrained(
        "openlm-research/open_llama_3b_v2",
        device_map="auto",
        torch_dtype=torch.float16,
    )

    device_string = PartialState().process_index

    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={'': device_string},
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = 'right'

    model, tokenizer = setup_chat_format(model, tokenizer)

    peft_config = LoraConfig(
        lora_alpha=128,
        lora_dropout=0.05,
        r=256,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    max_seq_length = 2048

    dataset = load_dataset("../data/")
    args = SFTConfig(output_dir="pretrained_chess_llm_ToC",
                     num_train_epochs=3,
                     per_device_train_batch_size=3,
                     gradient_accumulation_steps=2,
                     gradient_checkpointing=True,
                     optim="adamw_torch_fused",
                     logging_steps=10,
                     save_strategy="epoch",
                     learning_rate=2e-4,
                     bf16=True,
                     tf32=True,
                     max_grad_norm=0.3,
                     warmup_ratio=0.03,
                     lr_scheduler_type="constant",
                     push_to_hub=True,
                     report_to="wandb",
                     packing=True,
                     max_seq_length=max_seq_length,
                     dataset_kwargs={
                         "add_special_tokens": False,
                         "append_concat_token": False,
                     },
                     use_liger_kernel=True,
                     gradient_checkpointing_kwargs={'use_reentrant': False},
                     )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train_dataset.save_to_disk("data/Llama-3.2-3B-Instruct")
    trainer.train()

    trainer.save_model()
    args.distributed_state.wait_for_everyone()
    full_model = trainer.model.merge_and_unload()
    full_model.save_pretrained("pretrained_chess_llm_full")
    trainer.push_to_hub()
