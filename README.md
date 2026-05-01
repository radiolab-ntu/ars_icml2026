

# Answer-agreement Representation Shaping (ARS)

This repository contains the official implementation of our **ICML 2026** paper:

**Harnessing Reasoning Trajectories for Hallucination Detection via Answer-agreement Representation Shaping**  
[[arXiv]](https://arxiv.org/abs/2601.17467)

This codebase supports:
- Generating multiple LLM responses and extracting reasoning traces (CoT)
- Evaluating generations via **LLM-as-a-Judge** (Qwen3-32B)
- Generating noisy answer variants
- Judging answer agreement via **LLM-as-a-Judge** (semantic equivalence)
- Constructing ARS training data using answer-consistency supervision
- Training the proposed **Answer-agreement Representation Shaping (ARS)** module
- Evaluating hallucination detection using multiple embedding-based detectors

---

## Ads

If you are interested in embedding-based hallucination detection, you may also check our previous works:
- **HaloScope** (NeurIPS 2024 Spotlight) [[arXiv]](https://arxiv.org/abs/2409.17504)
- **TSV** (ICML 2025) [[arXiv]](https://arxiv.org/pdf/2503.01917)

---

## Requirements

Install all dependencies via:

```bash
pip install -r requirements.txt
````

---

## Models Preparation

Please download the **Qwen series** (e.g., `Qwen3-8B` and `Qwen3-14B`) and **DeepSeek-R1 series** (e.g., `DeepSeek-R1-Distill-Llama-8B` and `DeepSeek-R1-Distill-Qwen-14B`) from Hugging Face.

For **LLM-as-a-Judge**, the default judge model is `Qwen3-32B`.

We recommend creating a local directory to store all model checkpoints:

```bash
mkdir -p ./models
```

Then place all downloaded checkpoints inside `./models/`.

---

# Pipeline

## Step 1: Generate LLM Responses + Reasoning Traces + LLM-as-a-Judge Evaluation

This script generates multiple answers and automatically extracts reasoning traces (Chain-of-Thought).
It can optionally evaluate the generations using an LLM judge.

For **TruthfulQA**, run:

```bash
python ./generation_and_judge/generation_and_judge.py \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --dataset_name truthful_qa \
    --gen_mode greedy \
    --run_judge \
    --judge_model Qwen/Qwen3-32B \
    --max_samples 100 \
    --num_samples 5 \
    --output_dir ./outputs/generations
```

The output JSON file will be saved under:

```text
./outputs/generations/
```

---
# Step 2: Generate Noisy Variants + Judge Answer Agreement

We provide a script that generates multiple noisy variants for each sample by injecting noise into the **last token hidden state** during generation.
Then it optionally runs an LLM-based semantic judge to determine whether the variant answer is **semantically equivalent** to the original answer.

## 2.1 Generate Noisy Variants

```bash
python ./variant_generation_and_judge/generate_variants_and_judge.py \
    --input_path ./outputs/generations/exp1_truthfulqa.json \
    --model_name Qwen/Qwen3-8B \
    --num_variants 4 \
    --noise_scale 0.05 \
    --max_new_tokens 512 \
    --seed 42
```

This will output:

```text
./outputs/generations/exp1_truthfulqa_noisy.json
```

---

## 2.2 Run LLM-as-a-Judge for Semantic Consistency

To judge whether each variant answer is semantically equivalent to the original answer:

```bash
python ./variant_generation_and_judge/generate_variants_and_judge.py \
    --input_path ./outputs/generations/exp1_truthfulqa.json \
    --model_name Qwen/Qwen3-8B \
    --num_variants 4 \
    --noise_scale 0.05 \
    --max_new_tokens 512 \
    --seed 42 \
    --run_judge \
    --judge_model Qwen/Qwen3-8B
```

The judged output will be saved under:

```text
./result/Qwen_Qwen3-8B_exp1_truthfulqa_noisy.json
```


These agreement labels form the key supervision signal for ARS training.


---

## Step 3: Feature Extraction (Embedding Extraction)

This script extracts hidden-state embeddings from **all layers** of a pre-trained LLM.
It processes both the generated answers and reasoning traces.

To extract last-token embeddings across all layers:

```bash
python ./extracting/extract_embeddings.py \
    --data_file ./outputs/generations/exp1_truthfulqa.json \
    --model_name Qwen/Qwen3-8B \
    --save_dir ./outputs/embeddings/truthfulqa \
    --batch_size 1 \
    --seed 42
```

---

## Step 4: Extract Ground-truth Labels from Judge Responses

To extract the final grade (labels) from the judge outputs:

```bash
python ./extracting/extract_answer/extract_label.py \
    --input_json ./outputs/generations/exp1_truthfulqa.json \
    --output_npy ./outputs/labels/labels_truthfulqa.npy
```

---

## Step 5: ARS Training

Train ARS using the extracted embeddings:

```bash
python ./train_ars/ars_train.py train \
    --embeddings_dir ./outputs/embeddings/truthfulqa \
    --meta_file ./outputs/embeddings/truthfulqa/exp1_meta.npy \
    --save_dir ./outputs/ars_checkpoints \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-3
```

---

## Step 6: Extract Projected Features (Clean Subset)

After ARS training, extract the projected features of original samples (excluding variants):

```bash
python ./train_ars/ars_train.py extract \
    --original_root ./outputs/embeddings/truthfulqa \
    --projected_root ./outputs/ars_checkpoints \
    --meta_file ./outputs/embeddings/truthfulqa/exp1_meta.npy \
    --save_dir ./outputs/final_features
```

---

# Hallucination Detection

We evaluate hallucination detection performance using several embedding-based detectors.

---

## 1. Running CCS

To evaluate embeddings across all layers using CCS:

```bash
python ./detectors/train_ccs.py \
    --emb_root ./outputs/final_features \
    --seed 42 \
    --label_dir ./outputs/labels/labels_truthfulqa.npy \
    --output_dir ./outputs/results/ccs
```

---

## 2. Running Supervised Probing

To evaluate embeddings across all layers using supervised probing:

```bash
python ./detectors/train_probing.py \
    --emb_root ./outputs/final_features \
    --classifier logistic_regression \
    --seed 42 \
    --max_epochs 50 \
    --label_dir ./outputs/labels/labels_truthfulqa.npy \
    --output_dir ./outputs/results/probing
```

---

## 3. Running EigenScore

To evaluate embeddings across all layers using EigenScore:

```bash
python -u ./detectors/test_eigenscore.py \
    --emb_dir ./outputs/final_features \
    --seed 42 \
    --label_dir ./outputs/labels/labels_truthfulqa.npy \
    --output_dir ./outputs/results/eigenscore
```

---

## 4. Running HaloScope

To evaluate embeddings across all layers using HaloScope:

```bash
python ./detectors/train_haloscope.py \
    --emb_dir ./outputs/final_features \
    --seed 42 \
    --label_dir ./outputs/labels/labels_truthfulqa.npy \
    --output_dir ./outputs/results/haloscope
```

---

## Citation

If you find this repository useful, please consider citing:

```bibtex
@inproceedings{zhang2026harnessing,
  title={Harnessing Reasoning Trajectories for Hallucination Detection via Answer-agreement Representation Shaping},
  author={Jianxiong Zhang, Bing Guo, Yuming Jiang, Haobo Wang, Bo An, and Xuefeng Du},
  booktitle={International Conference on Machine Learning},
  year={2026}
}
```

---

## Additional Notes

This project can be sensitive to random seeds.
In all experiments reported in our paper, we fix the random seed to **42**. For more reliable reporting, we recommend averaging results across multiple random seeds.


