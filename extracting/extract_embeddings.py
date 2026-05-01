import os
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

def seed_everything(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = []
    for idx, item in enumerate(data):
        samples.append({
            "original_id": idx,
            "text": f"Question: {item['question']}\nReasoning: {item['trace']}\nAnswer: {item['answer']}",
            "answer": item["answer"],
            "is_consistent": True  
        })
        for var in item.get("variants", []):
            samples.append({
                "original_id": idx,
                "text": f"Question: {item['question']}\nReasoning: {item['trace']}\nAnswer: {var['generated_output']}",
                "answer": var["generated_output"],
                "is_consistent": var.get("is_consistent", False)
            })
    return samples

def get_last_token_embeddings_all_layers(text_list, tokenizer, model, batch_size=4):
    layer_embeddings = {}  
    num_layers = model.config.num_hidden_layers

    for l in range(1, num_layers + 1):  
        layer_embeddings[l] = []

    model_max_len = model.config.max_position_embeddings  

    for start_idx in tqdm(range(0, len(text_list), batch_size), desc="Extracting Layer Embeddings"):
        batch_texts = text_list[start_idx:start_idx + batch_size]

        batch_inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=model_max_len    
        )

        batch_inputs = {k: v.to(model.device) for k, v in batch_inputs.items()}
        attention_mask = batch_inputs["attention_mask"]

        with torch.no_grad():
            outputs = model(**batch_inputs, output_hidden_states=True)
            all_hidden = outputs.hidden_states  

        seq_lens = attention_mask.sum(dim=1)

        for i, seq_len in enumerate(seq_lens.tolist()):
            input_ids = batch_inputs["input_ids"][i]

            if attention_mask[i, seq_len - 1] == 0:
                print(f"[Warning] Sample {start_idx+i} last token is padding (unexpected).")

            if seq_len >= model_max_len:
                print(f"[Warning] Sample {start_idx+i} may be truncated by tokenizer/model.")

            if len(input_ids) > model_max_len:
                print(f"[Warning] Sample {start_idx+i} input length {len(input_ids)} > model limit {model_max_len}.")

        for layer_idx, hidden in enumerate(all_hidden):
            if layer_idx == 0:
                continue  

            last_token_emb = hidden[range(len(seq_lens)), seq_lens - 1, :]
            layer_embeddings[layer_idx].extend(last_token_emb.cpu().numpy())

        del batch_inputs, outputs, all_hidden
        torch.cuda.empty_cache()

    for k in layer_embeddings.keys():
        layer_embeddings[k] = np.array(layer_embeddings[k])

    return layer_embeddings



def main(args):
    seed_everything(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    prefix = os.path.splitext(os.path.basename(args.data_file))[0]

    samples = load_data(args.data_file)
    texts = [s["text"] for s in samples]
    original_ids = [s["original_id"] for s in samples]
    answers = [s["answer"] for s in samples]
    consistencies = [s["is_consistent"] for s in samples]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True
    )
    model.eval()


    layer_embeddings = get_last_token_embeddings_all_layers(texts, tokenizer, model, batch_size=args.batch_size)
    meta = np.array(list(zip(original_ids, answers, consistencies)), dtype=object)

    for layer_idx, emb in layer_embeddings.items():
        np.save(os.path.join(args.save_dir, f"{prefix}_layer{layer_idx}_embeddings.npy"), emb)

    np.save(os.path.join(args.save_dir, f"{prefix}_meta.npy"), meta)


    print(f"[Saved] Embeddings and meta info to {args.save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="data.json")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--save_dir", type=str, default="embeddings_icm")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
