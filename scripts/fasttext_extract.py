from pathlib import Path
import sys
from collections import Counter

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from Bio import SeqIO


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Datasets"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

POS_FASTA = DATA_DIR / "HighActivity.fasta"
NEG_FASTA = DATA_DIR / "LowActivity.fasta"


def read_fasta_sequences(fasta_path: Path):
    if not fasta_path.exists():
        raise FileNotFoundError(f"文件不存在: {fasta_path}")

    sequences = []
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        seq = str(record.seq).strip().upper()
        if seq:
            sequences.append(seq)

    if not sequences:
        raise ValueError(f"FASTA 里没有读到有效序列: {fasta_path}")

    return sequences


def load_dataset():
    pos = read_fasta_sequences(POS_FASTA)
    neg = read_fasta_sequences(NEG_FASTA)

    sequences = pos + neg
    labels = np.array([1] * len(pos) + [0] * len(neg), dtype=int)

    print(f"[INFO] positive: {len(pos)}")
    print(f"[INFO] negative: {len(neg)}")
    print(f"[INFO] total   : {len(sequences)}")

    return sequences, labels


def extract_features(peptide_sequences, vector_size=100, window=5, min_count=1, seed=42):
    tokenized_sequences = [list(seq) for seq in peptide_sequences]

    model = Word2Vec(
        sentences=tokenized_sequences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=0,
        workers=1,
        seed=seed,
        epochs=10,
    )

    vocabulary = list(model.wv.index_to_key)

    features = []
    for sequence in tokenized_sequences:
        bow = Counter(sequence)
        bow_vector = np.array([bow[token] for token in vocabulary], dtype=float)

        word2vec_vector = np.zeros(vector_size, dtype=float)
        valid_tokens = 0
        for token in sequence:
            if token in model.wv:
                word2vec_vector += model.wv[token]
                valid_tokens += 1

        if valid_tokens > 0:
            word2vec_vector /= valid_tokens

        combined_vector = np.concatenate([bow_vector, word2vec_vector], axis=0)
        features.append(combined_vector)

    features = np.asarray(features, dtype=float)
    return features, vocabulary


def main():
    sequences, labels = load_dataset()

    features, vocabulary = extract_features(
        sequences,
        vector_size=100,
        window=5,
        min_count=1,
        seed=42,
    )

    bow_dim = len(vocabulary)
    total_dim = features.shape[1]

    print(f"[INFO] vocabulary size: {bow_dim}")
    print(f"[INFO] feature shape  : {features.shape}")

    bow_columns = [f"bow_{token}" for token in vocabulary]
    word2vec_columns = [f"word2vec_{i}" for i in range(total_dim - bow_dim)]
    columns = bow_columns + word2vec_columns

    feature_df = pd.DataFrame(features, columns=columns)
    feature_path = OUT_DIR / f"Fasttext_all_{total_dim}.csv"
    feature_df.to_csv(feature_path, index=False)

    vocab_path = OUT_DIR / "Fasttext_vocab.txt"
    with open(vocab_path, "w", encoding="utf-8") as f:
        for token in vocabulary:
            f.write(token + "\n")

    seq_label_path = OUT_DIR / "seq_label_all.csv"
    if not seq_label_path.exists():
        pd.DataFrame({
            "sequence": sequences,
            "label": labels
        }).to_csv(seq_label_path, index=False)
        print(f"[OK] saved labels   : {seq_label_path}")
    else:
        print(f"[INFO] label file already exists, skipped overwrite: {seq_label_path}")

    print(f"[OK] saved features : {feature_path}")
    print(f"[OK] saved vocab    : {vocab_path}")


if __name__ == "__main__":
    main()