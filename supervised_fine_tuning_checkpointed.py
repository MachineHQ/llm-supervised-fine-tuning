import os
import torch
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, standardize_sharegpt, train_on_responses_only
from datasets import load_dataset
from huggingface_hub import create_repo, snapshot_download, HfApi
from huggingface_hub.errors import RepositoryNotFoundError
from pathlib import Path  # Needed for path management

# Move these imports to the bottom to avoid errors: https://github.com/unslothai/unsloth/issues/1264
from transformers import TrainingArguments, DataCollatorForSeq2Seq, TextStreamer
from trl import SFTTrainer

# Environment variables and settings
hf_token = os.environ.get("HF_TOKEN", None)  # Get a token at https://huggingface.co/settings/tokens
hf_target_repo = os.environ.get("HF_TARGET_REPO", None)  # The repo to save the model to.

# Ensure the target repo exists before resuming training
create_repo(f"{hf_target_repo}", token=hf_token, exist_ok=True, private=True)

source_model = os.environ.get("SOURCE_MODEL", "unsloth/Llama-3.2-3B-Instruct")
data_set_name = os.environ.get("DATA_SET", "mlabonne/FineTome-100k")

max_seq_length = int(os.environ.get("MAX_SEQ_LENGTH", 1024))  # Choose any! We auto support RoPE Scaling internally!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.
lora_rank = int(os.environ.get("LORA_RANK", 16))  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
lora_dropout = 0  # Supports any, but = 0 is optimized
learning_rate = float(os.environ.get("LEARNING_RATE", 2e-4))  # Suggested: 5e-6, 1e-5, 2e-5, 3e-5
max_steps = int(os.environ.get("MAX_STEPS", 100))  # Suggested: 100, 250, 500, 1000

# Load the base model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=source_model,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    token=os.environ.get("HF_TOKEN", None),
)

# Configure LoRA for the model
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=lora_rank,
    lora_dropout=lora_dropout,
    bias="none",  # Supports any, but "none" is optimized
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

# Update tokenizer with the chat template
tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3.1",
)

# Formatting function to process the dataset prompts
def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return {"text": texts}

# Load and preprocess the dataset
dataset = load_dataset(data_set_name, split="train")
dataset = standardize_sharegpt(dataset)
dataset = dataset.map(formatting_prompts_func, batched=True)

# Optionally inspect examples:
print(dataset[5]["conversations"])
print(dataset[5]["text"])

# Set up the SFTTrainer with TrainingArguments including checkpointing every 25 steps
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    dataset_num_proc=2,
    packing=False,  # Can make training 5x faster for short sequences.
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=max_steps,
        learning_rate=learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",  # Local output directory for checkpoints
        report_to="none",  # Use this for WandB etc.
        save_steps=25,  # Save checkpoint every 25 steps
        save_total_limit=3,  # Keep only the last 3 checkpoints
        save_strategy="steps",  # Use steps-based checkpointing
        push_to_hub=True,  # Push checkpoints to HF Hub
        hub_model_id=hf_target_repo,
        hub_strategy="checkpoint",  # This will push checkpoints
        hub_token=hf_token,
    ),
)

# Adjust the trainer for training on responses only
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
)

# Optional: inspect tokenized outputs
print(tokenizer.decode(trainer.train_dataset[5]["input_ids"]))
space = tokenizer(" ", add_special_tokens=False).input_ids[0]
print(tokenizer.decode([space if x == -100 else x for x in trainer.train_dataset[5]["labels"]]))

# ---------------------------
# Resume training logic below
# ---------------------------
def resume_training(trainer, repo_id, output_dir):
    """
    Attempts to resume training from the latest checkpoint in the given Hugging Face repository.
    Falls back to previous revisions if needed. If no valid checkpoint is found, starts fresh.
    """
    api = HfApi()

    def try_load_checkpoint(repo_path):
        checkpoint_dir = os.path.join(repo_path, "last-checkpoint")
        if os.path.exists(checkpoint_dir):
            print(f"Found last-checkpoint directory at {checkpoint_dir}")
            print(f"Checkpoint directory contents: {os.listdir(checkpoint_dir)}")
            state_file = os.path.join(checkpoint_dir, "trainer_state.json")
            if os.path.exists(state_file):
                import json
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    print(f"Resuming from step {state.get('global_step', 'unknown')}")
            try:
                stats = trainer.train(resume_from_checkpoint=checkpoint_dir)
            except ValueError as v:
                print(f"Failed to resume training: {str(v)}")
                return None
            return stats
        else:
            print(f"No last-checkpoint directory found in {repo_path}")
            return None

    try:
        # First, try to download the latest revision from the repo
        repo_path = snapshot_download(
            repo_id=repo_id,
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        print(f"Cloned repository to {repo_path}")
        print(f"Repository contents: {os.listdir(repo_path)}")
        stats = try_load_checkpoint(repo_path)
        if stats is not None:
            return stats

        # If the latest revision did not contain a valid checkpoint, try previous revisions
        repo_versions = api.list_repo_commits(repo_id)
        print(f"Found {len(repo_versions)} versions of repository {repo_id}")
        for commit in repo_versions[1:]:  # Skip the first since we already tried it
            print(f"Checking revision {commit.commit_id}")
            try:
                repo_path = snapshot_download(
                    repo_id=repo_id,
                    revision=commit.commit_id,
                    local_dir=output_dir,
                    local_dir_use_symlinks=False
                )
                print(f"Downloaded revision contents: {os.listdir(repo_path)}")
                stats = try_load_checkpoint(repo_path)
                if stats is not None:
                    return stats
            except Exception as e:
                print(f"Failed to check revision {commit.commit_id}: {str(e)}")
                continue

        print("No valid checkpoints found in repository history")
    except RepositoryNotFoundError:
        print(f"Repository {repo_id} not found")
    except Exception as e:
        print(f"Error accessing repository: {str(e)}")

    # If we get here, no checkpoint worked or repo doesn't exist – start training from scratch
    print("Starting training from scratch")
    stats = trainer.train()
    return stats

# Print GPU memory stats before training starts
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# Resume training using our new helper
trainer_stats = resume_training(trainer, hf_target_repo, output_dir="outputs")

# ---------------------------
# Post-training GPU and timing stats
# ---------------------------
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# ---------------------------
# Inference section remains unchanged
# ---------------------------
tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3.1",
)
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

messages = [
    {"role": "user", "content": "Continue the fibonnaci sequence: 1, 1, 2, 3, 5, 8,"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,  # Must add for generation
    return_tensors="pt",
).to("cuda")

outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=64,
    use_cache=True,
    temperature=1.5,
    min_p=0.1
)
print(tokenizer.batch_decode(outputs))

FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

messages = [
    {"role": "user", "content": "Continue the fibonnaci sequence: 1, 1, 2, 3, 5, 8,"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,  # Must add for generation
    return_tensors="pt",
).to("cuda")

text_streamer = TextStreamer(tokenizer, skip_prompt=True)
_ = model.generate(
    input_ids=inputs,
    streamer=text_streamer,
    max_new_tokens=128,
    use_cache=True,
    temperature=1.5,
    min_p=0.1
)

# Save the final model locally
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

# --- Inference using the saved LoRA adapters ---
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="lora_model",  # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

messages = [
    {"role": "user", "content": "Describe a tall tower in the capital of France."},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,  # Must add for generation
    return_tensors="pt",
).to("cuda")

text_streamer = TextStreamer(tokenizer, skip_prompt=True)
_ = model.generate(
    input_ids=inputs,
    streamer=text_streamer,
    max_new_tokens=128,
    use_cache=True,
    temperature=1.5,
    min_p=0.1
)

# Finally, push the final model to the hub
model.push_to_hub_merged(f"{hf_target_repo}", tokenizer, save_method="merged_16bit", token=hf_token)
