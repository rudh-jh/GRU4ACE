from pathlib import Path
import numpy as np
import pandas as pd
from Bio import SeqIO
from scipy.io import savemat


ROOT = Path(r"E:\MYS\GRU4ACE")
POS_FASTA = ROOT / "Datasets" / "HighActivity.fasta"
NEG_FASTA = ROOT / "Datasets" / "LowActivity.fasta"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
PAD_IDX = 20
MAX_LEN = 21


def read_fasta_sequences(fasta_path: Path) -> list[str]:
    seqs = []
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        seqs.append(str(record.seq).strip().upper())
    return seqs


def exchange_matrix(protein: str, max_len: int = MAX_LEN) -> np.ndarray:
    protein = protein.upper()
    idxs = [AA_TO_IDX.get(aa, PAD_IDX) for aa in protein[:max_len]]
    idxs += [PAD_IDX] * (max_len - len(idxs))
    return np.array(idxs, dtype=np.int32)


def build_bpf_features(sequences: list[str]) -> np.ndarray:
    encoded = [exchange_matrix(seq) for seq in sequences]
    matrix = np.vstack(encoded)

    vectors = []
    for row in matrix:
        vector = []
        for value in row:
            feature = np.zeros(21, dtype=np.int8)
            feature[int(value)] = 1
            vector.extend(feature)
        vectors.append(vector)

    return matrix, np.array(vectors, dtype=np.int8)


def main():
    pos_seqs = read_fasta_sequences(POS_FASTA)
    neg_seqs = read_fasta_sequences(NEG_FASTA)

    sequence = np.array(pos_seqs + neg_seqs, dtype=object)
    label = np.array([1] * len(pos_seqs) + [0] * len(neg_seqs), dtype=int)

    print(f"positive: {len(pos_seqs)}")
    print(f"negative: {len(neg_seqs)}")
    print(f"total: {len(sequence)}")

    matrix, matrix_two = build_bpf_features(sequence.tolist())

    print("matrix shape:", matrix.shape)
    print("BPF feature shape:", matrix_two.shape)

    savemat(OUT_DIR / "BPF_all.mat", {
        "matrix_two": matrix_two,
        "label": label,
        "sequence": sequence
    })

    pd.DataFrame(matrix_two).to_csv(OUT_DIR / "BPF_all_441.csv", index=False)
    pd.DataFrame({
        "sequence": sequence,
        "label": label
    }).to_csv(OUT_DIR / "seq_label_all.csv", index=False)

    print("Saved:")
    print(OUT_DIR / "BPF_all.mat")
    print(OUT_DIR / "BPF_all_441.csv")
    print(OUT_DIR / "seq_label_all.csv")


if __name__ == "__main__":
    main()