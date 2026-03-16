from pathlib import Path
import os

# ---- 离线与兼容性设置 ----
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from transformers import BertTokenizer, BertModel


# =========================
# 路径配置
# =========================
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Datasets"
OUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "hf_models" / "bert-base-uncased"

OUT_DIR.mkdir(parents=True, exist_ok=True)

POS_FASTA = DATA_DIR / "HighActivity.fasta"
NEG_FASTA = DATA_DIR / "LowActivity.fasta"

# 3070 Ti 先从 16 开始；显存不够就改成 8 或 4
BATCH_SIZE = 16

# True=只跑前10条冒烟测试；False=跑全量
SMOKE_TEST = False
SMOKE_N = 10


# =========================
# 数据读取
# =========================
def read_fasta_sequences(fasta_path: Path):
    if not fasta_path.exists():
        raise FileNotFoundError(f"文件不存在: {fasta_path}")

    ids = []
    sequences = []

    for record in SeqIO.parse(str(fasta_path), "fasta"):
        seq = str(record.seq).strip().upper()
        if seq:
            ids.append(record.id)
            sequences.append(seq)

    if not sequences:
        raise ValueError(f"FASTA 里没有有效序列: {fasta_path}")

    return ids, sequences


def load_dataset():
    pos_ids, pos_seqs = read_fasta_sequences(POS_FASTA)
    neg_ids, neg_seqs = read_fasta_sequences(NEG_FASTA)

    ids = pos_ids + neg_ids
    sequences = pos_seqs + neg_seqs
    labels = np.array([1] * len(pos_seqs) + [0] * len(neg_seqs), dtype=np.int64)

    print(f"[INFO] positive: {len(pos_seqs)}")
    print(f"[INFO] negative: {len(neg_seqs)}")
    print(f"[INFO] total   : {len(sequences)}")

    return ids, sequences, labels


# =========================
# 工具函数
# =========================
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def batch_iter(ids, seqs, batch_size):
    for i in range(0, len(seqs), batch_size):
        yield ids[i:i + batch_size], seqs[i:i + batch_size]


def validate_local_model_dir(model_dir: Path):
    required_always = [
        "config.json",
        "vocab.txt",
        "tokenizer_config.json",
    ]

    missing = [x for x in required_always if not (model_dir / x).exists()]
    has_bin = (model_dir / "pytorch_model.bin").exists()
    has_safe = (model_dir / "model.safetensors").exists()

    if not model_dir.exists():
        raise FileNotFoundError(
            f"本地模型目录不存在: {model_dir}\n"
            f"请先手动下载 bert-base-uncased 并放到这个目录。"
        )

    if missing:
        raise FileNotFoundError(
            "本地模型目录缺少以下文件:\n"
            + "\n".join(f" - {x}" for x in missing)
            + f"\n当前目录: {model_dir}"
        )

    if not has_bin and not has_safe:
        raise FileNotFoundError(
            f"{model_dir} 中未找到 pytorch_model.bin 或 model.safetensors"
        )


# =========================
# BERT 特征提取
# =========================
def extract_bert_features(ids, sequences, batch_size=BATCH_SIZE):
    validate_local_model_dir(MODEL_DIR)

    device = get_device()
    print(f"[INFO] device: {device}")
    if device.type == "cuda":
        print(f"[INFO] gpu name: {torch.cuda.get_device_name(0)}")

    print(f"[INFO] loading tokenizer from local dir: {MODEL_DIR}")
    tokenizer = BertTokenizer.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
    )
    print("[INFO] tokenizer loaded")

    print(f"[INFO] loading model from local dir: {MODEL_DIR}")
    model = BertModel.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    model = model.to(device)
    model.eval()
    print("[INFO] model loaded")

    all_features = []
    kept_ids = []

    total = len(sequences)
    processed = 0

    for batch_ids, batch_seqs in batch_iter(ids, sequences, batch_size):
        encoded_input = tokenizer(
            batch_seqs,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

        with torch.no_grad():
            outputs = model(**encoded_input)

        hidden_states = outputs.last_hidden_state
        pooled_output = torch.mean(hidden_states, dim=1)  # 与原 notebook 保持一致

        batch_features = pooled_output.detach().float().cpu().numpy()

        all_features.append(batch_features)
        kept_ids.extend(batch_ids)

        processed += len(batch_seqs)
        print(f"[INFO] processed {processed}/{total}")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    features = np.concatenate(all_features, axis=0).astype(np.float32)
    return kept_ids, features


# =========================
# 主流程
# =========================
def main():
    ids, sequences, labels = load_dataset()

    if SMOKE_TEST:
        ids = ids[:SMOKE_N]
        sequences = sequences[:SMOKE_N]
        labels = labels[:SMOKE_N]
        print(f"[INFO] smoke test enabled: only first {SMOKE_N} sequences")

    kept_ids, features = extract_bert_features(
        ids=ids,
        sequences=sequences,
        batch_size=BATCH_SIZE,
    )

    print(f"[INFO] feature shape: {features.shape}")

    feature_dim = features.shape[1]
    feature_cols = [f"BERT_{i}" for i in range(feature_dim)]

    feature_csv = OUT_DIR / f"BERT_all_{feature_dim}.csv"
    feature_npy = OUT_DIR / f"BERT_all_{feature_dim}.npy"
    meta_csv = OUT_DIR / "BERT_ids.csv"

    pd.DataFrame(features, columns=feature_cols).to_csv(feature_csv, index=False)
    np.save(feature_npy, features)

    pd.DataFrame({
        "id": kept_ids,
        "sequence": sequences,
        "label": labels,
    }).to_csv(meta_csv, index=False)

    seq_label_path = OUT_DIR / "seq_label_all.csv"
    if not seq_label_path.exists():
        pd.DataFrame({
            "sequence": sequences,
            "label": labels,
        }).to_csv(seq_label_path, index=False)
        print(f"[OK] saved labels   : {seq_label_path}")
    else:
        print(f"[INFO] label file already exists, skipped overwrite: {seq_label_path}")

    print(f"[OK] saved features : {feature_csv}")
    print(f"[OK] saved features : {feature_npy}")
    print(f"[OK] saved meta     : {meta_csv}")


if __name__ == "__main__":
    main()