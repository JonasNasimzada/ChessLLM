import torch
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, LlamaTokenizer
from trl import setup_chat_format, SFTTrainer, SFTConfig, clone_chat_template
from accelerate import PartialState

if __name__ == "__main__":
    login(
        token="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",
        add_to_git_credential=True
    )

    model_id = "openlm-research/open_llama_3b_v2"

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
        quantization_config=bnb_config,
        low_cpu_mem_usage=True
    )

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
        task_type="CAUSAL_LM"
    )

    max_seq_length = 2048  # max sequence length for model and packing of the dataset

    dataset = load_dataset("./data/")
    args = SFTConfig(output_dir="pretrained_chess_llm_ToC",  # directory to save and repository id
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
                     report_to="wandb",  # report metrics to wandb
                     packing=True,
                     max_seq_length=max_seq_length,
                     dataset_kwargs={
                         "add_special_tokens": False,  # We template with special tokens
                         "append_concat_token": False,  # No need to add additional separator token
                     },
                     use_liger_kernel=True,
                     gradient_checkpointing_kwargs={'use_reentrant': False}

                     )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    trainer.train()

    trainer.accelerator.state.fsdp_plugin.set_state_dict_type('FULL_STATE_DICT')
    trainer.model.config.use_cache = True
    trainer.save_model()
    args.distributed_state.wait_for_everyone()
    tokenizer.save_pretrained("pretrained_chess_llm_ToC")
    trainer.push_to_hub()
    # save model

    # Merge LoRA weights into the base model and save the full merged model

    # `trainer.model` is a PeftModel with LoRA adapters
    # full_model = trainer.model.merge_and_unload()
    # full_model.save_pretrained("pretrained_chess_llm_full")
