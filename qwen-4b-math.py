# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "unsloth",
#   "vllm==0.15.1",
#   "transformers==4.56.2",
#   "trl==0.22.2",
#   "datasets",
#   "pandas",
#   "numpy",
#   "torchvision",
#   "bitsandbytes",
#   "xformers",
#   "torchao>=0.16.0",
#   "safetensors",
#   "wandb",
# ]
# ///
import os
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"

from unsloth import FastLanguageModel

import re
import gc
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from trl import SFTTrainer, SFTConfig, GRPOConfig, GRPOTrainer
from vllm import SamplingParams
from safetensors import safe_open


max_seq_length = 2048
lora_rank = 32

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-4B-Base",
    max_seq_length=max_seq_length,
    load_in_4bit=False,
    fast_inference=True,
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.9,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=lora_rank * 2,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

reasoning_start = "<start_working_out>"
reasoning_end = "<end_working_out>"
solution_start = "<SOLUTION>"
solution_end = "</SOLUTION>"

system_prompt = (
    f"You are given a problem.\n"
    f"Think about the problem and provide your working out.\n"
    f"Place it between {reasoning_start} and {reasoning_end}.\n"
    f"Then, provide your solution between {solution_start}{solution_end}"
)

chat_template = (
    "{% if messages[0]['role'] == 'system' %}"
        "{{ messages[0]['content'] + eos_token }}"
        "{% set loop_messages = messages[1:] %}"
    "{% else %}"
        "{{ '{system_prompt}' + eos_token }}"
        "{% set loop_messages = messages %}"
    "{% endif %}"
    "{% for message in loop_messages %}"
        "{% if message['role'] == 'user' %}"
            "{{ message['content'] }}"
        "{% elif message['role'] == 'assistant' %}"
            "{{ message['content'] + eos_token }}"
        "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '{reasoning_start}' }}"
    "{% endif %}"
)
chat_template = (
    chat_template
    .replace("'{system_prompt}'", f"'{system_prompt}'")
    .replace("'{reasoning_start}'", f"'{reasoning_start}'")
)
tokenizer.chat_template = chat_template


# Pre fine-tune for formatting
dataset = load_dataset("unsloth/OpenMathReasoning-mini", split="cot")
dataset = dataset.to_pandas()[["expected_answer", "problem", "generated_solution"]]
is_number = pd.to_numeric(pd.Series(dataset["expected_answer"]), errors="coerce").notnull()
dataset = dataset.iloc[np.where(is_number)[0]]


def format_dataset(x):
    expected_answer = x["expected_answer"]
    problem = x["problem"]
    thoughts = x["generated_solution"].replace("<think>", "").replace("</think>", "").strip()
    final_prompt = (
        reasoning_start + thoughts + reasoning_end
        + solution_start + expected_answer + solution_end
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": final_prompt},
    ]


dataset["Messages"] = dataset.apply(format_dataset, axis=1)
dataset["N"] = dataset["Messages"].apply(lambda x: len(tokenizer.apply_chat_template(x)))
dataset = dataset.loc[dataset["N"] <= max_seq_length / 2].copy()
dataset["text"] = tokenizer.apply_chat_template(dataset["Messages"].values.tolist(), tokenize=False)
dataset = Dataset.from_pandas(dataset)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        warmup_steps=5,
        num_train_epochs=2,
        learning_rate=2e-4,
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="wandb",
    ),
)
trainer.train()

del dataset
torch.cuda.empty_cache()
gc.collect()


# GRPO
dataset = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split="train")
dataset = dataset.map(lambda x: {
    "prompt": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": x["prompt"]},
    ],
    "answer": x["solution"],
})

solution_end_regex = r"</SOLUTION>[\s]{0,}" + "(?:" + re.escape(tokenizer.eos_token) + ")?"
match_format = re.compile(
    rf"{reasoning_end}.*?{solution_start}(.+?){solution_end_regex}[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)
match_numbers = re.compile(
    solution_start + r".*?[\s]{0,}([-]?[\d\.\,]{1,})",
    flags=re.MULTILINE | re.DOTALL,
)


def match_format_exactly(completions, **kwargs):
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        scores.append(3.0 if match_format.search(response) is not None else 0)
    return scores


def match_format_approximately(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        score += 0.5 if response.count(reasoning_end) == 1 else -1.0
        score += 0.5 if response.count(solution_start) == 1 else -1.0
        score += 0.5 if response.count(solution_end) == 1 else -1.0
        scores.append(score)
    return scores


def check_answer(prompts, completions, answer, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [
        (guess.group(1) if (guess := match_format.search(r)) is not None else None)
        for r in responses
    ]
    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        score = 0
        if guess is None:
            scores.append(-2.0)
            continue
        if guess == true_answer:
            score += 5.0
        elif guess.strip() == true_answer.strip():
            score += 3.5
        else:
            try:
                ratio = float(guess) / float(true_answer)
                if 0.9 <= ratio <= 1.1:
                    score += 2.0
                elif 0.8 <= ratio <= 1.2:
                    score += 1.5
                else:
                    score -= 2.5
            except Exception:
                score -= 4.5
        scores.append(score)
    return scores


PRINTED_TIMES = 0
PRINT_EVERY_STEPS = 5


def check_numbers(prompts, completions, answer, **kwargs):
    global PRINTED_TIMES
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [
        (guess.group(1) if (guess := match_numbers.search(r)) is not None else None)
        for r in responses
    ]
    if PRINTED_TIMES % PRINT_EVERY_STEPS == 0:
        print(
            "*" * 20 + f"Question:\n{question}",
            f"\nAnswer:\n{answer[0]}",
            f"\nResponse:\n{responses[0]}",
            f"\nExtracted:\n{extracted_responses[0]}",
        )
    PRINTED_TIMES += 1

    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        if guess is None:
            scores.append(-2.5)
            continue
        try:
            true_answer = float(true_answer.strip())
            guess = float(guess.strip().replace(",", ""))
            scores.append(3.5 if guess == true_answer else -1.5)
        except Exception:
            scores.append(0)
    return scores


tokenized = dataset.map(
    lambda x: {"tokens": tokenizer.apply_chat_template(x["prompt"], add_generation_prompt=True, tokenize=True)},
    batched=True,
)
tokenized = tokenized.map(lambda x: {"L": len(x["tokens"])})
maximum_length = int(np.quantile(tokenized["L"], 0.9))
print("Max Length = ", maximum_length)
dataset = dataset.select(np.where(np.array(tokenized["L"]) <= maximum_length)[0])
del tokenized

max_prompt_length = maximum_length + 1
max_completion_length = max_seq_length - max_prompt_length

vllm_sampling_params = SamplingParams(
    min_p=0.1,
    top_p=1.0,
    top_k=-1,
    seed=3407,
    stop=[tokenizer.eos_token],
    include_stop_str_in_output=True,
)

training_args = GRPOConfig(
    vllm_sampling_params=vllm_sampling_params,
    temperature=1.0,
    learning_rate=5e-6,
    weight_decay=0.001,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    optim="adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_generations=8,
    max_prompt_length=max_prompt_length,
    max_completion_length=max_completion_length,
    max_steps=100,
    save_steps=100,
    report_to="wandb",
    output_dir="outputs",
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        match_format_exactly,
        match_format_approximately,
        check_answer,
        check_numbers,
    ],
    args=training_args,
    train_dataset=dataset,
)
trainer.train()

model.save_lora("grpo_saved_lora")

with safe_open("grpo_saved_lora/adapter_model.safetensors", framework="pt") as f:
    for key in f.keys():
        tensor = f.get_tensor(key)
        n_zeros = (tensor == 0).sum() / tensor.numel()
        assert n_zeros.item() != tensor.numel()

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What is the sqrt of 101?"},
]
text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
sampling_params = SamplingParams(temperature=1.0, top_k=50, max_tokens=2048)
output = model.fast_generate(
    text,
    sampling_params=sampling_params,
    lora_request=model.load_lora("grpo_saved_lora"),
)[0].outputs[0].text
print(output)
