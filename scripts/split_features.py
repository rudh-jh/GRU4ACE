from pathlib import Path
import json
import numpy as np
import pandas as pd


ROOT = Path(r"E:\MYS\GRU4ACE")
OUT_DIR = ROOT / "outputs"
SPLIT_DIR = OUT_DIR / "split_seed42"
FEATURE_SPLIT_DIR = SPLIT_DIR / "features"
FEATURE_SPLIT_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LABEL_FILE = OUT_DIR / "seq_label_all.csv"
TRAIN_IDX_FILE = SPLIT_DIR / "train_idx.npy"
TEST_IDX_FILE = SPLIT_DIR / "test_idx.npy"

# 一次性登记全部特征
FEATURE_FILES = {
    "BPF": OUT_DIR / "BPF_all_441.csv",
    "FEGS": OUT_DIR / "FEGS_all_578.csv",
    "Fasttext": OUT_DIR / "Fasttext_all_120.csv",
    "ProtT5": OUT_DIR / "ProtT5_all_1024.npy",
    "BERT": OUT_DIR / "BERT_all_768.npy",
    "ESM2": OUT_DIR / "ESM2_all_320.npy",
}

EXPECTED_DIMS = {
    "BPF": 441,
    "FEGS": 578,
    "Fasttext": 120,
    "ProtT5": 1024,
    "BERT": 768,
    "ESM2": 320,
}


def load_split_indices():
    train_idx = np.load(TRAIN_IDX_FILE)
    test_idx = np.load(TEST_IDX_FILE)
    return train_idx, test_idx


def load_feature_matrix(feature_file: Path):
    if not feature_file.exists():
        raise FileNotFoundError(f"找不到文件: {feature_file}")

    suffix = feature_file.suffix.lower()

    if suffix == ".csv":
        X = pd.read_csv(feature_file).to_numpy()
    elif suffix == ".npy":
        X = np.load(feature_file)
    else:
        raise ValueError(f"暂不支持的特征文件格式: {feature_file}")

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    return X


def split_one_feature(
    feature_name: str,
    feature_file: Path,
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
):
    X_np = load_feature_matrix(feature_file)

    if len(X_np) != len(labels):
        raise ValueError(
            f"{feature_name} 行数与标签数不一致: "
            f"features={len(X_np)}, labels={len(labels)}"
        )

    X_train = X_np[train_idx]
    X_test = X_np[test_idx]
    y_train = labels[train_idx]
    y_test = labels[test_idx]

    expected_dim = EXPECTED_DIMS.get(feature_name)
    if expected_dim is not None and X_np.shape[1] != expected_dim:
        print(
            f"[WARN] {feature_name} 实际维度为 {X_np.shape[1]}，"
            f"与预期 {expected_dim} 不一致"
        )

    print(f"\n[{feature_name}]")
    print(f"source file  : {feature_file.name}")
    print(f"full shape   : {X_np.shape}")
    print(f"train shape  : {X_train.shape}")
    print(f"test shape   : {X_test.shape}")
    print(f"train labels : pos={(y_train == 1).sum()}, neg={(y_train == 0).sum()}")
    print(f"test labels  : pos={(y_test == 1).sum()}, neg={(y_test == 0).sum()}")

    # 保存 csv
    pd.DataFrame(X_train).to_csv(FEATURE_SPLIT_DIR / f"{feature_name}_X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv(FEATURE_SPLIT_DIR / f"{feature_name}_X_test.csv", index=False)
    pd.DataFrame({"label": y_train}).to_csv(FEATURE_SPLIT_DIR / f"{feature_name}_y_train.csv", index=False)
    pd.DataFrame({"label": y_test}).to_csv(FEATURE_SPLIT_DIR / f"{feature_name}_y_test.csv", index=False)

    # 保存 numpy
    np.save(FEATURE_SPLIT_DIR / f"{feature_name}_X_train.npy", X_train)
    np.save(FEATURE_SPLIT_DIR / f"{feature_name}_X_test.npy", X_test)
    np.save(FEATURE_SPLIT_DIR / f"{feature_name}_y_train.npy", y_train)
    np.save(FEATURE_SPLIT_DIR / f"{feature_name}_y_test.npy", y_test)

    print("saved:")
    print(FEATURE_SPLIT_DIR / f"{feature_name}_X_train.csv")
    print(FEATURE_SPLIT_DIR / f"{feature_name}_X_test.csv")
    print(FEATURE_SPLIT_DIR / f"{feature_name}_y_train.csv")
    print(FEATURE_SPLIT_DIR / f"{feature_name}_y_test.csv")
    print(FEATURE_SPLIT_DIR / f"{feature_name}_X_train.npy")
    print(FEATURE_SPLIT_DIR / f"{feature_name}_X_test.npy")
    print(FEATURE_SPLIT_DIR / f"{feature_name}_y_train.npy")
    print(FEATURE_SPLIT_DIR / f"{feature_name}_y_test.npy")

    return {
        "feature_name": feature_name,
        "source_file": feature_file.name,
        "full_rows": int(X_np.shape[0]),
        "full_cols": int(X_np.shape[1]),
        "train_rows": int(X_train.shape[0]),
        "train_cols": int(X_train.shape[1]),
        "test_rows": int(X_test.shape[0]),
        "test_cols": int(X_test.shape[1]),
    }


def main():
    if not SEQ_LABEL_FILE.exists():
        raise FileNotFoundError(f"找不到标签文件: {SEQ_LABEL_FILE}")
    if not TRAIN_IDX_FILE.exists() or not TEST_IDX_FILE.exists():
        raise FileNotFoundError("找不到 train/test 索引，请先运行 split_dataset.py")

    seq_label_df = pd.read_csv(SEQ_LABEL_FILE)
    if "label" not in seq_label_df.columns:
        raise ValueError("seq_label_all.csv 必须包含 label 列")

    labels = seq_label_df["label"].to_numpy()
    train_idx, test_idx = load_split_indices()

    print(f"total samples: {len(labels)}")
    print(f"train size   : {len(train_idx)}")
    print(f"test size    : {len(test_idx)}")

    summaries = []
    for feature_name, feature_file in FEATURE_FILES.items():
        summary = split_one_feature(feature_name, feature_file, labels, train_idx, test_idx)
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(FEATURE_SPLIT_DIR / "feature_split_summary.csv", index=False)

    with open(FEATURE_SPLIT_DIR / "feature_split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print("All split feature files saved to:")
    print(FEATURE_SPLIT_DIR)
    print(FEATURE_SPLIT_DIR / "feature_split_summary.csv")
    print(FEATURE_SPLIT_DIR / "feature_split_summary.json")


if __name__ == "__main__":
    main()