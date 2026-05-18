# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "unsloth",
#   "vllm==0.15.1",
#   "transformers==4.57.0",
#   "trl==0.26.2",
#   "datasets>=3.4.1,<4.0.0",
#   "pandas",
#   "numpy",
#   "torchvision",
#   "bitsandbytes",
#   "xformers",
#   "torchao>=0.16.0",
#   "safetensors",
#   "wandb",
#   "huggingface_hub>=0.34.0",
#   "hf_transfer",
#   "sentencepiece",
#   "protobuf",
#   "peft",
#   "accelerate",
#   "triton",
#   "cut_cross_entropy",
#   "unsloth_zoo",
# ]
# ///
import os
os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")

from unsloth import FastVisionModel

import re
import torch
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from safetensors import safe_open


max_seq_length = 16384
lora_rank = 16

model, tokenizer = FastVisionModel.from_pretrained(
    model_name="unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=False,
    gpu_memory_utilization=float(os.environ.get("GPU_MEM_UTIL", 0.8)),
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=lora_rank,
    lora_alpha=lora_rank,
    lora_dropout=0,
    bias="none",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
    use_gradient_checkpointing="unsloth",
)


REASONING_START = "<REASONING>"
REASONING_END = "</REASONING>"
SOLUTION_START = "<SOLUTION>"
SOLUTION_END = "</SOLUTION>"


dataset = load_dataset("AI4Math/MathVista", split="testmini")


def is_numeric_answer(example):
    try:
        float(example["answer"])
        return True
    except Exception:
        return False


dataset = dataset.filter(is_numeric_answer)


def resize_images(example):
    image = example["decoded_image"]
    example["decoded_image"] = image.resize((512, 512))
    return example


dataset = dataset.map(resize_images)


def convert_to_rgb(example):
    image = example["decoded_image"]
    if image.mode != "RGB":
        example["decoded_image"] = image.convert("RGB")
    return example


dataset = dataset.map(convert_to_rgb)


def make_conversation(example):
    text_content = (
        f"{example['question']}. Also first provide your reasoning or working out"
        f" on how you would go about solving the question between {REASONING_START} and {REASONING_END}"
        f" and then your final answer between {SOLUTION_START} and (put a single float here) {SOLUTION_END}"
    )
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text_content},
            ],
        },
    ]
    return {"prompt": prompt, "image": example["decoded_image"], "answer": example["answer"]}


train_dataset = dataset.map(make_conversation)
train_dataset = train_dataset.remove_columns("image")
train_dataset = train_dataset.rename_column("decoded_image", "image")


def formatting_reward_func(completions, **kwargs):
    thinking_pattern = f"{REASONING_START}(.*?){REASONING_END}"
    answer_pattern = f"{SOLUTION_START}(.*?){SOLUTION_END}"

    scores = []
    for completion in completions:
        if isinstance(completion, list):
            completion = completion[0]["content"] if completion else ""
        score = 0
        thinking_matches = re.findall(thinking_pattern, completion, re.DOTALL)
        answer_matches = re.findall(answer_pattern, completion, re.DOTALL)
        if len(thinking_matches) == 1:
            score += 1.0
        if len(answer_matches) == 1:
            score += 1.0

        if len(completion) != 0:
            removal = completion.replace("addCriterion", "").replace("\n", "")
            if (len(completion) - len(removal)) / len(completion) >= 0.5:
                score -= 2.0

        scores.append(score)
    return scores


def correctness_reward_func(prompts, completions, answer, **kwargs):
    answer_pattern = f"{SOLUTION_START}(.*?){SOLUTION_END}"
    completions = [(c[0]["content"] if c else "") if isinstance(c, list) else c for c in completions]
    responses = [re.findall(answer_pattern, completion, re.DOTALL) for completion in completions]
    q = prompts[0]
    print("-" * 20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:{completions[0]}")
    return [
        2.0 if len(r) == 1 and a == r[0].replace("\n", "") else 0.0
        for r, a in zip(responses, answer)
    ]


training_args = GRPOConfig(
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    logging_steps=1,
    log_completions=False,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_generations=2,
    max_prompt_length=1024,
    max_completion_length=1024,
    num_train_epochs=0.5,
    save_steps=60,
    max_grad_norm=0.1,
    report_to="wandb",
    output_dir="outputs",
    importance_sampling_level="sequence",
    mask_truncated_completions=False,
    loss_type="dr_grpo",
)

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    reward_funcs=[
        formatting_reward_func,
        correctness_reward_func,
    ],
    train_dataset=train_dataset,
)

trainer.train()

model.save_pretrained("grpo_lora")
tokenizer.save_pretrained("grpo_lora")

with safe_open("grpo_lora/adapter_model.safetensors", framework="pt") as f:
    for key in f.keys():
        tensor = f.get_tensor(key)
        n_zeros = (tensor == 0).sum() / tensor.numel()
        assert n_zeros.item() != tensor.numel()
