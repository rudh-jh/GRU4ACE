from pathlib import Path
import os
import re

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
import esm
from Bio import SeqIO


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Datasets"
OUT_DIR = ROOT / "outputs"
MODEL_FILE = ROOT / "hf_models" / "esm2_t6_8M_UR50D.pt"

OUT_DIR.mkdir(parents=True, exist_ok=True)

POS_FASTA = DATA_DIR / "HighActivity.fasta"
NEG_FASTA = DATA_DIR / "LowActivity.fasta"

BATCH_SIZE = 16
SMOKE_TEST = False
SMOKE_N = 10


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


def clean_sequence(seq: str) -> str:
    seq = seq.upper()
    seq = re.sub(r"[UZOB]", "X", seq)
    return seq


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def batch_iter(ids, seqs, batch_size):
    for i in range(0, len(seqs), batch_size):
        yield ids[i:i + batch_size], seqs[i:i + batch_size]


def validate_local_model_file(model_file: Path):
    if not model_file.exists():
        raise FileNotFoundError(
            f"本地模型文件不存在: {model_file}\n"
            f"请先手动下载 esm2_t6_8M_UR50D.pt 并放到该路径。"
        )


def extract_esm2_features(ids, sequences, batch_size=BATCH_SIZE):
    validate_local_model_file(MODEL_FILE)

    device = get_device()
    print(f"[INFO] device: {device}")
    if device.type == "cuda":
        print(f"[INFO] gpu name: {torch.cuda.get_device_name(0)}")

    print(f"[INFO] loading ESM2 model from local file: {MODEL_FILE}")
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(MODEL_FILE))
    batch_converter = alphabet.get_batch_converter()
    model = model.to(device)
    model.eval()
    print("[INFO] model loaded")

    all_features = []
    kept_ids = []

    total = len(sequences)
    processed = 0

    for batch_ids, batch_seqs in batch_iter(ids, sequences, batch_size):
        cleaned_batch = [clean_sequence(seq) for seq in batch_seqs]
        data = list(zip(batch_ids, cleaned_batch))

        _, _, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        with torch.no_grad():
            results = model(
                batch_tokens,
                repr_layers=[6],
                return_contacts=False,
            )

        token_representations = results["representations"][6]

        for i, tokens_len in enumerate(batch_lens):
            tokens_len = int(tokens_len.item())
            seq_repr = token_representations[i, 1:tokens_len - 1].mean(0)
            emb = seq_repr.detach().float().cpu().numpy()

            all_features.append(emb)
            kept_ids.append(batch_ids[i])

        processed += len(batch_seqs)
        print(f"[INFO] processed {processed}/{total}")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    features = np.asarray(all_features, dtype=np.float32)
    return kept_ids, features


def main():
    ids, sequences, labels = load_dataset()

    if SMOKE_TEST:
        ids = ids[:SMOKE_N]
        sequences = sequences[:SMOKE_N]
        labels = labels[:SMOKE_N]
        print(f"[INFO] smoke test enabled: only first {SMOKE_N} sequences")

    kept_ids, features = extract_esm2_features(
        ids=ids,
        sequences=sequences,
        batch_size=BATCH_SIZE,
    )

    print(f"[INFO] feature shape: {features.shape}")

    feature_dim = features.shape[1]
    feature_cols = [f"ESM2_{i}" for i in range(feature_dim)]

    feature_csv = OUT_DIR / f"ESM2_all_{feature_dim}.csv"
    feature_npy = OUT_DIR / f"ESM2_all_{feature_dim}.npy"
    meta_csv = OUT_DIR / "ESM2_ids.csv"

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