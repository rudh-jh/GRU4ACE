from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eig
from scipy.io import savemat
from Bio import SeqIO


ROOT = Path(r"E:\MYS\GRU4ACE")
POS_FASTA = ROOT / "Datasets" / "HighActivity.fasta"
NEG_FASTA = ROOT / "Datasets" / "LowActivity.fasta"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def read_fasta_sequences(fasta_path: Path) -> list[str]:
    seqs = []
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        seqs.append(str(record.seq).strip().upper())
    return seqs


def SAD(seq: str, amino_acids: str):
    len_seq = len(seq)
    len_a = len(amino_acids)
    AAC = np.zeros(len_a, dtype=float)
    DPC = np.zeros((len_a, len_a), dtype=float)

    for i in range(len_a):
        AAC[i] = seq.count(amino_acids[i]) / len_seq

    if len_seq > 1:
        for i in range(len_a):
            for j in range(len_a):
                count = sum(
                    1
                    for k in range(len_seq - 1)
                    if seq[k] == amino_acids[i] and seq[k + 1] == amino_acids[j]
                )
                DPC[i, j] = count / (len_seq - 1)
    else:
        DPC = np.zeros((20, 20), dtype=float)

    return AAC, DPC


def ME(W: np.ndarray) -> float:
    if W.ndim == 1:
        W = W.reshape(-1, 1)

    W = W[1:, :]
    D = pdist(W)
    E = squareform(D)

    x = W.shape[0]
    sdist = np.zeros((x, x), dtype=float)

    for i in range(x):
        for j in range(i, x):
            if j - i == 1:
                sdist[i, j] = E[i, j]
            elif j - i > 1:
                sdist[i, j] = sdist[i, j - 1] + E[j - 1, j]

    sd = sdist + sdist.T
    sdd = sd + np.eye(x)
    L = E / sdd

    eigenvalues = eig(L, right=False)
    largest_eigval = np.max(eigenvalues.real)

    return float(largest_eigval / x)


def coordinate():
    P = np.zeros((20, 3), dtype=float)
    V = np.zeros((20, 20, 3), dtype=float)

    for i in range(20):
        P[i] = [np.cos(i * 2 * np.pi / 20), np.sin(i * 2 * np.pi / 20), 1]

    for i in range(20):
        for j in range(20):
            V[i, j] = P[i] + 0.25 * (P[j] - P[i])

    return P, V


def GRS(seq: str, P: np.ndarray, V: np.ndarray, M: np.ndarray):
    l_seq = len(seq)
    k = M.shape[0]

    g = []
    for j in range(k):
        c = np.zeros(3, dtype=float)
        d = np.zeros(3, dtype=float)
        y = np.zeros(20, dtype=bool)

        for i in range(l_seq):
            x = (seq[i] == M[j, :])

            if i == 0:
                c = c + x.dot(P)
            elif np.all(x == 0):
                d = d * (i - 1) / i
                c = c + np.array([0, 0, 1], dtype=float) + d
            elif np.all(y == 0):
                d = d * (i - 1) / i
                c = c + x.dot(P) + d
            else:
                prev_idx = np.where(y == 1)[0][0]
                curr_idx = np.where(x == 1)[0][0]
                d = d * (i - 1) / i + V[prev_idx, curr_idx] / i
                c = c + x.dot(P) + d

            y = x

        g.append(c)

    return np.array(g, dtype=float)


def FEGS_from_sequences(sequences: list[str]) -> np.ndarray:
    P, V = coordinate()
    l = len(sequences)

    EL = np.zeros((l, 158), dtype=float)
    FA = np.zeros((l, 20), dtype=float)
    FD = np.zeros((l, 400), dtype=float)

    char = "ARNDCQEGHILKMFPSTWYV"
    M = np.array([list(char)] * 158)

    for i, seq in enumerate(sequences):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"Processing {i + 1}/{l} ...")

        g_p = GRS(seq, P, V, M)

        for u in range(158):
            EL[i, u] = ME(g_p[u])

        AAC, DPC = SAD(seq, char)
        FA[i, :] = AAC
        FD[i, :] = DPC.flatten()

    FV = np.hstack([EL, FA, FD])
    return FV


def main():
    pos_seqs = read_fasta_sequences(POS_FASTA)
    neg_seqs = read_fasta_sequences(NEG_FASTA)

    sequences = pos_seqs + neg_seqs
    labels = np.array([1] * len(pos_seqs) + [0] * len(neg_seqs), dtype=int)

    print(f"positive: {len(pos_seqs)}")
    print(f"negative: {len(neg_seqs)}")
    print(f"total: {len(sequences)}")

    features = FEGS_from_sequences(sequences)

    print("FEGS feature shape:", features.shape)

    pd.DataFrame(features).to_csv(OUT_DIR / "FEGS_all_578.csv", index=False)
    pd.DataFrame({
        "sequence": sequences,
        "label": labels
    }).to_csv(OUT_DIR / "seq_label_all.csv", index=False)

    savemat(OUT_DIR / "FEGS_all.mat", {
        "features": features,
        "label": labels,
        "sequence": np.array(sequences, dtype=object)
    })

    print("Saved:")
    print(OUT_DIR / "FEGS_all_578.csv")
    print(OUT_DIR / "FEGS_all.mat")
    print(OUT_DIR / "seq_label_all.csv")


if __name__ == "__main__":
    main()