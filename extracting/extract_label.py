import json
import re
import os
import argparse
import numpy as np
from collections import Counter


def process_json_and_save(input_file: str, output_file: str = None):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    label_counter = Counter()

    for item in data:
        acc_text = item.get("model_judged_acc", "")

        match = re.findall(r"\b([ABC])\b", acc_text.strip())
        if match:
            final_grade = match[-1]
        else:
            final_grade = "C"  

        mapping = {
            "A": "CORRECT",
            "B": "INCORRECT",
            "C": "NOT_ATTEMPTED",
        }
        mapped_label = mapping.get(final_grade, "NOT_ATTEMPTED")

        item["final_grade"] = final_grade
        item["mapped_label"] = mapped_label

        label_counter[mapped_label] += 1

    if output_file is None:
        name, ext = os.path.splitext(input_file)
        output_file = f"{name}_processed{ext}"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    total = sum(label_counter.values())
    for label in ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"]:
        count = label_counter[label]
        ratio = count / total * 100 if total > 0 else 0
        print(f"  {label}: {count} ({ratio:.2f}%)")

    return output_file 

def json_to_npy_labels(json_file, output_npy):
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"can not find files: {json_file}")

    with open(json_file, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    labels = []
    for item in data_list:
        raw_label = item.get("mapped_label", "").strip().upper()
        if raw_label == "CORRECT":
            labels.append(0)
        elif raw_label in ["INCORRECT", "NOT_ATTEMPTED"]:
            labels.append(1)
        else:
            raise ValueError(f"wrong labels: {raw_label}")

    labels_array = np.array(labels, dtype=np.int64)
    np.save(output_npy, labels_array)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--processed_json",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output_npy",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    processed_json = process_json_and_save(
        args.input_json, args.processed_json
    )
    json_to_npy_labels(processed_json, args.output_npy)
