import os
import time
import json
import argparse
import re
from copy import deepcopy
from typing import Dict

import torch
import numpy as np
import random
from loguru import logger
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def extract_after_assistant(text: str) -> str:
    pattern = r"<\uFF5CAssistant\uFF5C>(.*)$"
    m = re.search(pattern, text, flags=re.DOTALL)
    if m:
        return m.group(1).lstrip()
    return text


def split_by_sentence_ratio(text: str, ratio: float = 0.8):
    sentences = re.split(r"(?<=[。！？.!?])", text)
    sentences = [s for s in sentences if s.strip()]
    if len(sentences) == 0:
        return "<think></think>", text
    split_idx = max(1, int(len(sentences) * ratio))
    trace = "".join(sentences[:split_idx]).strip()
    answer = "".join(sentences[split_idx:]).strip()
    trace = f"<think>{trace}</think>"
    return trace, answer


def build_generation_kwargs(args):
    if args.gen_mode == "greedy":
        return dict(do_sample=False, num_beams=1)
    if args.gen_mode == "sample":
        return dict(do_sample=True, num_beams=1, temperature=0.5, top_p=0.95, top_k=20)
    if args.gen_mode == "beam":
        return dict(do_sample=False, num_beams=args.num_beam)
    raise ValueError("Unknown generation mode")


def generate_trace_qwen(args, dataset, save_path) -> Dict:
    model_name = args.model_name
    generated_qta = []

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    model.eval()

    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            generated_qta = json.load(f)
        start_idx = len(generated_qta)
        logger.info(f"Resuming from index {start_idx}")
    else:
        start_idx = 0

    gen_kwargs = build_generation_kwargs(args)

    for i in tqdm(range(start_idx, len(dataset)), initial=start_idx):
        question = dataset[i]['question']
        messages = [{"role": "user", "content": f"Answer the question concisely. Q: {question}"}]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        if args.gen_mode == "sample":
            samples = []

            for k in range(args.num_samples):

                with torch.no_grad():
                    outputs = model.generate(
                        **model_inputs,
                        max_new_tokens=4096,
                        return_dict_in_generate=True,
                        **gen_kwargs
                    )

                output_ids = outputs.sequences[0][len(model_inputs.input_ids[0]):].tolist()

                try:
                    index = len(output_ids) - output_ids[::-1].index(151668)
                    trace = tokenizer.decode(output_ids[:index], skip_special_tokens=True)
                    answer = tokenizer.decode(output_ids[index:], skip_special_tokens=True)
                except ValueError:
                    raw = tokenizer.decode(output_ids, skip_special_tokens=True)
                    trace, answer = split_by_sentence_ratio(raw)

                samples.append({
                    "trace": trace,
                    "answer": answer
                })

            generated_qta.append({
                "question": question,
                "samples": samples,
                "original_data": dataset[i]
            })

        else:
            with torch.no_grad():
                outputs = model.generate(
                    **model_inputs,
                    max_new_tokens=4096,
                    return_dict_in_generate=True,
                    **gen_kwargs
                )

            output_ids = outputs.sequences[0][len(model_inputs.input_ids[0]):].tolist()

            try:
                index = len(output_ids) - output_ids[::-1].index(151668)
                trace = tokenizer.decode(output_ids[:index], skip_special_tokens=True)
                answer = tokenizer.decode(output_ids[index:], skip_special_tokens=True)
            except ValueError:
                raw = tokenizer.decode(output_ids, skip_special_tokens=True)
                trace, answer = split_by_sentence_ratio(raw)

            generated_qta.append({
                "question": question,
                "trace": trace,
                "answer": answer,
                "original_data": dataset[i]
            })


        with open(save_path, "w") as f:
            json.dump(generated_qta, f, indent=2, ensure_ascii=False)

    return generated_qta


def generate_trace_deepseek_r1(args, dataset, save_path) -> Dict:
    model_name = args.model_name
    generated_qta = []

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    model.eval()

    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            generated_qta = json.load(f)
        start_idx = len(generated_qta)
    else:
        start_idx = 0

    gen_kwargs = build_generation_kwargs(args)

    for i in tqdm(range(start_idx, len(dataset)), initial=start_idx):
        question = dataset[i]['question']
        messages = [{"role": "user", "content": f"Answer the question concisely. Q: {question}"}]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        if args.gen_mode == "sample":
            samples = []
            for k in range(args.num_samples):
                with torch.no_grad():
                    outputs = model.generate(
                        **model_inputs,
                        max_new_tokens=4096,
                        return_dict_in_generate=True,
                        do_sample=True,
                        num_beams=1,
                        temperature=0.5,
                        top_p=0.95,
                        top_k=20
                    )

                gen_ids = outputs.sequences[0][len(model_inputs.input_ids[0]):]
                assistant_text = tokenizer.decode(gen_ids, skip_special_tokens=False)

                if "</think>" in assistant_text:
                    idx = assistant_text.index("</think>") + len("</think>")
                    trace = assistant_text[:idx].strip()
                    if not trace.startswith("<think>"):
                        trace = "<think>" + trace
                    answer = assistant_text[idx:].strip()
                else:
                    trace, answer = split_by_sentence_ratio(assistant_text)

                samples.append({"trace": trace, "answer": answer})

            generated_qta.append({
                "question": question,
                "samples": samples,
                "original_data": dataset[i]
            })
        else:
            with torch.no_grad():
                outputs = model.generate(
                    **model_inputs,
                    max_new_tokens=4096,
                    return_dict_in_generate=True,
                    **gen_kwargs
                )

            gen_ids = outputs.sequences[0][len(model_inputs.input_ids[0]):]
            assistant_text = tokenizer.decode(gen_ids, skip_special_tokens=False)

            if "</think>" in assistant_text:
                idx = assistant_text.index("</think>") + len("</think>")
                trace = assistant_text[:idx].strip()
                if not trace.startswith("<think>"):
                    trace = "<think>" + trace
                answer = assistant_text[idx:].strip()
            else:
                trace, answer = split_by_sentence_ratio(assistant_text)

            generated_qta.append({
                "question": question,
                "trace": trace,
                "answer": answer,
                "original_data": dataset[i]
            })

        with open(save_path, "w") as f:
            json.dump(generated_qta, f, indent=2, ensure_ascii=False)

    return generated_qta


def judge_answer(args, model_answers: Dict, save_path: str) -> Dict:
    tokenizer = AutoTokenizer.from_pretrained(args.judge_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.judge_model,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model.eval()

    with open("prompt_LLM_as_a_judge.txt", "r") as f:
        GRADER_TEMPLATE = f.read()

    accuracy = []
    start_idx = 0
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            accuracy = json.load(f)
        start_idx = len(accuracy)

    for i in tqdm(range(start_idx, len(model_answers))):
        d = model_answers[i]
        messages = [{
            "role": "user",
            "content": GRADER_TEMPLATE.format(
                question=d["question"],
                targets=d["original_data"]["correct_answers"],
                predicted_answer=d["answer"]
            )
        }]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        answer_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        current = deepcopy(d)
        current["model_judged_acc"] = answer_text.strip()
        accuracy.append(current)

        with open(save_path, "w") as f:
            json.dump(accuracy, f, indent=2)

    return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="truthful_qa")
    parser.add_argument("--gen_mode", type=str, default="greedy")
    parser.add_argument("--run_judge", action="store_true")
    parser.add_argument("--judge_model", type=str, default="Qwen/Qwen3-32B")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Only run on the first N samples (for debugging)"
    )
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--num_beam", type=int, default=1)



    args = parser.parse_args()
    seed_everything(42)

    logger.add("pipeline.log", rotation="1 GB")

    # load dataset
    if args.dataset_name == "truthful_qa":
        dataset = load_dataset("truthful_qa", 'generation')['validation']
    elif args.dataset_name == 'triviaqa':
        dataset = load_dataset("trivia_qa", "rc.nocontext", split="validation")
        id_mem = set()

        def remove_dups(batch):
            if batch['question_id'][0] in id_mem:
                return {_: [] for _ in batch.keys()}
            id_mem.add(batch['question_id'][0])
            return batch

        dataset = dataset.map(remove_dups, batch_size=1, batched=True, load_from_cache_file=False)
    elif args.dataset_name == 'gsm8k':
        dataset = load_dataset("gsm8k", "main")['train']  

    elif args.dataset_name == 'math500':
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test") 
    
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))


    model_id = args.model_name.split("/")[-1]
    gen_file = f"exp1_{model_id}_{args.dataset_name}.json"

    model_id_lower = model_id.lower()

    if "deepseek" in model_id_lower:
        generate_trace_deepseek_r1(args, dataset, gen_file)
    elif "qwen" in model_id_lower:
        generate_trace_qwen(args, dataset, gen_file)
    else:
        raise ValueError(
            f"Unsupported model: {model_id}. "
            "Only Qwen-* and DeepSeek-* models are supported."
        )


    if args.run_judge:
        with open(gen_file, "r") as f:
            data = json.load(f)
        judge_file = f"exp2_{model_id}_{args.dataset_name}.json"
        judge_answer(args, data, judge_file)


if __name__ == "__main__":
    main()
