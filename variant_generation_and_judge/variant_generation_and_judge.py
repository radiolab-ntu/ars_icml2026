import json
import random
import argparse
import os
import torch
from typing import List, Optional
import torch.nn.functional as F
from contextlib import contextmanager
from tqdm.auto import tqdm

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception:
    AutoTokenizer = None
    AutoModelForCausalLM = None

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    print(f"[Info] Global random seed set to {seed}")

@torch.no_grad()
def generate_first_token_noisy(trace, model, tokenizer,
                               num_variants=4,
                               noise_scale=0.05,
                               max_new_tokens=512, temperature=0.7,
                               seed: int = 42):
    device = model.device

    messages = [
        {"role": "user",
         "content": f"Continue reasoning from the given trajectory and generate the next part of the answer. Trajectory:\n{trace}"}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True,
                                         enable_thinking=False)
    print(f"[Debug] Prompt text length: {len(text)}")

    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    print(f"[Debug] input_ids shape: {input_ids.shape}, attention_mask shape: {attention_mask.shape}")

    last_block = None
    last_block_name = None
    for name, module in model.named_modules():
        if any(k in name for k in ["decoder.layers", "layers"]):
            last_block = module
            last_block_name = name
    assert last_block is not None, "can not find the transformer block"
    print(f"[Info] Hook will be added to: {last_block_name}")

    variants = []

    for v_idx in range(num_variants):
        local_seed = seed + v_idx * 13
        torch.manual_seed(local_seed)

        hook_handle = []
        injected = [False]

        def hook_last_token_noise(module, input, output):
            if not injected[0]:
                print(f"[Hook Triggered] Module: {module.__class__.__name__}, Output shape: {output.shape}")
                output[:, -1:, :] += noise_scale * torch.randn_like(output[:, -1:, :])
                injected[0] = True
                print(f"[Hook] Noise injected for variant {v_idx+1}")
            hook_handle[0].remove()

        hook_handle.append(last_block.register_forward_hook(hook_last_token_noise))

        print(f"[Info] Generating variant {v_idx+1}/{num_variants} ...")
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        new_tokens = generated[:, input_ids.shape[1]:]
        gen_text = tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()
        print(f"[Debug] Variant {v_idx+1} generated text (first 100 chars): {gen_text[:100]}...")

        variants.append(gen_text)

    return variants

def process_json(
    input_path: str,
    sample_size: int = 200,
    num_variants: int = 4,
    seed: int = 42,
    tokenizer_name: Optional[str] = None,
    model_name: Optional[str] = None,
    noise_scale: float = 0.05,
    max_new_tokens: int = 512,
):
    set_seed(seed)

    if AutoTokenizer is None or AutoModelForCausalLM is None:
        raise RuntimeError("transformers are not install")

    print(f"[Info] Loading tokenizer and model: {tokenizer_name or model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name or tokenizer_name,
        torch_dtype=torch.float16
    ).cuda()
    model.eval()

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sampled_items = data
    if sample_size is not None:
        sampled_items = data[:sample_size]

    sample_n = len(sampled_items)

    base = os.path.splitext(os.path.basename(input_path))[0]
    output_name = f"{base}_noisy.json"

    with open(output_name, "w", encoding="utf-8") as f:
        f.write("[\n")

    for i, item in enumerate(sampled_items):
        trace = item.get("trace", "")

        try:
            results = generate_first_token_noisy(
                trace=trace,
                num_variants=num_variants,
                tokenizer=tokenizer,
                model=model,
                noise_scale=noise_scale,
                seed=seed,
                max_new_tokens=max_new_tokens
            )

            saved_item = {k: item[k] for k in ("question", "trace", "answer", "original_data") if k in item}
            saved_item["variants"] = [
                {"generated_output": gen} for gen in results
            ]

            with open(output_name, "a", encoding="utf-8") as f:
                json.dump(saved_item, f, ensure_ascii=False, indent=2)
                if i < sample_n - 1:
                    f.write(",\n")
                else:
                    f.write("\n")

        except Exception as e:
            print(f"[Error] {i}: {e}")
            continue

    with open(output_name, "a", encoding="utf-8") as f:
        f.write("]\n")

    print(f"\n[Done] All processed samples saved to {output_name}")
    return output_name

import re

def load_model(model_name="Qwen/Qwen3-8B"):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    return tokenizer, model

CONSISTENCY_PROMPT = """You are an expert semantic judge specializing in factual reasoning and truthfulness evaluation.
You will be given two answers (A and B) to the same factual question.
Your task is to determine whether these two answers are *semantically equivalent* —
that is, whether they convey the same factual meaning or final conclusion,
even if their wording or phrasing differs.

Two answers are considered **equivalent** if:
- They state the same fact, claim, or conclusion.
- Differences are only in style, explanation, or level of detail.
- Minor paraphrasing or rewording does not change the meaning.

Two answers are considered **not equivalent** if:
- They express conflicting or different factual claims.
- One is vague, incomplete, or contradicts the other.
- One is “Cannot determine” or “Unknown,” while the other gives a definite factual claim.

---

### Example (One-shot)

Question:
Can humans breathe underwater without any equipment?

Answer A:
No, humans cannot breathe underwater without help.

Answer B:
Humans can’t survive underwater without diving gear.

Judgment:
Yes

---

Question:
Can humans breathe underwater without any equipment?

Answer A:
Yes, some people can train themselves to do so.

Answer B:
No, it’s impossible without breathing apparatus.

Judgment:
No

---

Now judge the following pair:

Question:
{question}

Answer A:
{original}

Answer B:
{variant}

Are these two answers semantically equivalent?
Please output exactly one token: "Yes" or "No".
Do not include any punctuation, explanation, or additional words.
"""

def judge_consistency(tokenizer, model, ques, original, variant, debug=False):
    prompt = CONSISTENCY_PROMPT.format(
        question=ques.strip(),
        original=original.strip(),
        variant=variant.strip()
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                temperature=0.0,
                use_cache=True
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()

        text_lower = text.lower()
        text_lower = re.sub(r"(answer:)?\s*yes\s+or\s+no[\?:]?", "", text_lower, flags=re.IGNORECASE).strip()
        match = re.search(r"\b(yes|no)\b", text_lower)
        if not match:
            if debug:
                print(f"[can not find the Yes/No]：{repr(text)}")
            return None, text

        result = match.group(1) == "yes"

        return result

    except Exception as e:
        print(f"[Error]：{e}")
        return None, f"[ERROR] {e}"

def process_judge(input_path, model_name="Qwen/Qwen3-8B", debug=False):
    tokenizer, model = load_model(model_name)
    result_dir = "result"
    os.makedirs(result_dir, exist_ok=True)

    input_filename = os.path.basename(input_path)
    safe_model_name = model_name.replace("/", "_")
    output_path = os.path.join(result_dir, f"{safe_model_name}_{input_filename}")

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    total_variants = sum(len(item["variants"]) for item in data)
    done_count = sum(
        1 for item in data for v in item["variants"]
        if isinstance(v, dict) and "is_consistent" in v
    )

    progress = tqdm(total=total_variants, desc="Checking consistency", initial=done_count)

    for i, item in enumerate(data):
        ques = item.get("question", "")
        orig_answer = item.get("answer", "")

        for idx, var_ans in enumerate(item.get("variants", [])):
            if isinstance(var_ans, dict) and "is_consistent" in var_ans:
                continue

            generated_answer = var_ans.get("generated_output") if isinstance(var_ans, dict) else var_ans

            try:
                result = judge_consistency(tokenizer, model,
                                                       ques=ques,
                                                       original=orig_answer,
                                                       variant=generated_answer,
                                                       debug=debug)
                item["variants"][idx] = {**var_ans, "is_consistent": result}
            except Exception as e:
                item["variants"][idx] = {
                    "generated_output": var_ans,
                    "is_consistent": None
                }

            progress.update(1)

        if (i + 1) % 10 == 0:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    progress.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--num_variants", type=int, default=4)
    parser.add_argument("--noise_scale", type=float, default=0.05)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_judge", action="store_true")
    parser.add_argument("--judge_model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--sample_size", type=int, default=None)
    args = parser.parse_args()

    variant_file = process_json(
        input_path=args.input_path,
        sample_size=args.sample_size,
        num_variants=args.num_variants,
        seed=args.seed,
        tokenizer_name=args.tokenizer_name,
        model_name=args.model_name,
        noise_scale=args.noise_scale,
        max_new_tokens=args.max_new_tokens
    )

    if args.run_judge:
        process_judge(variant_file, model_name=args.judge_model)
