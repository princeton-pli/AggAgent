import argparse
import base64
import hashlib
import json
import os
import shutil
import urllib.request

import pandas


DATA_DIR = "data"


def derive_key(password: str, length: int) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(password.encode())
    key = hasher.digest()
    return key * (length // len(key)) + key[: length % len(key)]


def decrypt(ciphertext_b64: str, password: str) -> str:
    encrypted = base64.b64decode(ciphertext_b64)
    key = derive_key(password, len(encrypted))
    decrypted = bytes(a ^ b for a, b in zip(encrypted, key))
    return decrypted.decode()


def download_browsecomp():
    print("Downloading BrowseComp dataset...")
    df = pandas.read_csv(
        "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"
    )

    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, "browsecomp.jsonl")

    if os.path.exists(out_path):
        print(f"Already exists, skipping: {out_path}")
        return

    with open(out_path, "w") as f:
        for _, row in df.iterrows():
            canary = row["canary"]
            problem = decrypt(row["problem"], canary)
            answer = decrypt(row["answer"], canary)
            record = {"problem": problem, "answer": answer}
            f.write(json.dumps(record) + "\n")

    print(f"Saved {len(df)} examples to {out_path}")


def download_deepsearchqa():
    print("Downloading DeepSearchQA dataset...")
    out_path = os.path.join(DATA_DIR, "deepsearchqa.jsonl")

    if os.path.exists(out_path):
        print(f"Already exists, skipping: {out_path}")
        return

    from datasets import load_dataset

    dataset = load_dataset("google/deepsearchqa", "deepsearchqa")
    os.makedirs(DATA_DIR, exist_ok=True)

    with open(out_path, "w") as f:
        for example in dataset["eval"]:
            record = {
                "problem": example["problem"],
                "answer": example["answer"],
                "problem_category": example["problem_category"],
                "answer_type": example["answer_type"],
            }
            f.write(json.dumps(record) + "\n")

    print(f"Saved {len(dataset['eval'])} examples to {out_path}")


def download_healthbench():
    print("Downloading HealthBench (hard) dataset...")
    out_path = os.path.join(DATA_DIR, "healthbench.jsonl")

    if os.path.exists(out_path):
        print(f"Already exists, skipping: {out_path}")
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    url = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl"
    urllib.request.urlretrieve(url, out_path)

    with open(out_path) as f:
        count = sum(1 for _ in f)
    print(f"Saved {count} examples to {out_path}")


def download_researchrubrics():
    print("Downloading ResearchRubrics dataset...")
    out_path = os.path.join(DATA_DIR, "researchrubrics.jsonl")

    if os.path.exists(out_path):
        print(f"Already exists, skipping: {out_path}")
        return

    from datasets import load_dataset

    dataset = load_dataset("ScaleAI/researchrubrics")
    os.makedirs(DATA_DIR, exist_ok=True)

    with open(out_path, "w") as f:
        for example in dataset["train"]:
            record = {
                "prompt": example["prompt"],
                "sample_id": example["sample_id"],
                "domain": example["domain"],
                "conceptual_breadth": example["conceptual_breadth"],
                "logical_nesting": example["logical_nesting"],
                "exploration": example["exploration"],
                "rubrics": example["rubrics"],
            }
            f.write(json.dumps(record) + "\n")

    print(f"Saved {len(dataset['train'])} examples to {out_path}")


def download_browsecomp_plus():
    from huggingface_hub import snapshot_download

    # Create browsecomp-plus.jsonl as a copy of browsecomp.jsonl
    src_path = os.path.join(DATA_DIR, "browsecomp.jsonl")
    out_path = os.path.join(DATA_DIR, "browsecomp-plus.jsonl")
    if os.path.exists(out_path):
        print(f"Already exists, skipping: {out_path}")
    else:
        if not os.path.exists(src_path):
            download_browsecomp()
        shutil.copy(src_path, out_path)
        print(f"Copied {src_path} -> {out_path}")

    browsecomp_plus_dir = os.path.join(DATA_DIR, "browsecomp-plus")
    os.makedirs(browsecomp_plus_dir, exist_ok=True)

    # Download retrieval indexes (qwen3-embedding-8b)
    indexes_dir = os.path.join(browsecomp_plus_dir, "indexes")
    if os.path.exists(indexes_dir):
        print(f"Indexes already exist, skipping: {indexes_dir}")
    else:
        print("Downloading BrowseComp+ indexes...")
        snapshot_download(
            repo_id="Tevatron/browsecomp-plus-indexes",
            repo_type="dataset",
            allow_patterns="qwen3-embedding-8b/*",
            local_dir=indexes_dir,
        )
        print(f"Saved indexes to {indexes_dir}")

    # Download corpus (kept as parquet)
    corpus_dir = os.path.join(browsecomp_plus_dir, "corpus")
    if os.path.exists(corpus_dir):
        print(f"Corpus already exists, skipping: {corpus_dir}")
    else:
        print("Downloading BrowseComp+ corpus...")
        snapshot_download(
            repo_id="Tevatron/browsecomp-plus-corpus",
            repo_type="dataset",
            local_dir=corpus_dir,
        )
        print(f"Saved corpus to {corpus_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--browsecomp-plus", action="store_true",
                        help="Also download BrowseComp+ indexes and corpus")
    args = parser.parse_args()

    download_browsecomp()
    if args.browsecomp_plus:
        download_browsecomp_plus()
    else:
        download_deepsearchqa()
        download_healthbench()
        download_researchrubrics()

    
