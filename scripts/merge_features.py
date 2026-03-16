from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(r"E:\MYS\GRU4ACE")
OUT_DIR = ROOT / "outputs"
SPLIT_DIR = OUT_DIR / "split_seed42"
FEATURE_SPLIT_DIR = SPLIT_DIR / "features"
MERGE_DIR = SPLIT_DIR / "merged"
MERGE_DIR.mkdir(parents=True, exist_ok=True)

# 想合并哪些特征，就在这里改
FEATURES_TO_MERGE = [
    "BPF",
    "FEGS",
    "Fasttext",
    "ProtT5",
    "BERT",
    "ESM2",
]

# 可选：用于打印核对维度，不影响运行
EXPECTED_DIMS = {
    "BPF": 441,
    "FEGS": 578,
    "Fasttext": 120,
    "ProtT5": 1024,
    "BERT": 768,
    "ESM2": 320,
}


def load_array(feature_name: str, split_name: str):
    """
    优先读取 .npy；如果没有，再读 .csv
    split_name: X_train / X_test / y_train / y_test
    """
    npy_file = FEATURE_SPLIT_DIR / f"{feature_name}_{split_name}.npy"
    csv_file = FEATURE_SPLIT_DIR / f"{feature_name}_{split_name}.csv"

    if npy_file.exists():
        arr = np.load(npy_file)
        return arr

    if csv_file.exists():
        df = pd.read_csv(csv_file)
        if split_name.startswith("y_"):
            if "label" in df.columns:
                return df["label"].to_numpy()
            if df.shape[1] == 1:
                return df.iloc[:, 0].to_numpy()
            raise ValueError(f"{csv_file} 不是标准标签文件，未找到 label 列")
        return df.to_numpy()

    raise FileNotFoundError(f"找不到文件: {npy_file} 或 {csv_file}")


def load_feature_split(feature_name: str):
    x_train = load_array(feature_name, "X_train")
    x_test = load_array(feature_name, "X_test")
    y_train = load_array(feature_name, "y_train")
    y_test = load_array(feature_name, "y_test")

    # 保证 X 是二维
    if x_train.ndim == 1:
        x_train = x_train.reshape(-1, 1)
    if x_test.ndim == 1:
        x_test = x_test.reshape(-1, 1)

    # 保证 y 是一维
    y_train = np.asarray(y_train).reshape(-1)
    y_test = np.asarray(y_test).reshape(-1)

    return x_train, x_test, y_train, y_test


def check_labels_consistent(all_y_train: list[np.ndarray], all_y_test: list[np.ndarray]):
    base_y_train = all_y_train[0]
    base_y_test = all_y_test[0]

    for i in range(1, len(all_y_train)):
        if not np.array_equal(base_y_train, all_y_train[i]):
            raise ValueError("不同特征的 y_train 不一致，说明切分顺序有问题")
        if not np.array_equal(base_y_test, all_y_test[i]):
            raise ValueError("不同特征的 y_test 不一致，说明切分顺序有问题")

    return base_y_train, base_y_test


def main():
    if not FEATURES_TO_MERGE:
        raise ValueError("FEATURES_TO_MERGE 不能为空")

    train_parts = []
    test_parts = []
    all_y_train = []
    all_y_test = []

    print("Merging features:")
    print(", ".join(FEATURES_TO_MERGE))

    for feature_name in FEATURES_TO_MERGE:
        x_train, x_test, y_train, y_test = load_feature_split(feature_name)

        print(f"\n[{feature_name}]")
        print(f"X_train shape: {x_train.shape}")
        print(f"X_test  shape: {x_test.shape}")

        if feature_name in EXPECTED_DIMS:
            expected_dim = EXPECTED_DIMS[feature_name]
            if x_train.shape[1] != expected_dim:
                print(
                    f"[WARN] {feature_name} 训练集维度为 {x_train.shape[1]}，"
                    f"与预期 {expected_dim} 不一致"
                )

        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError(f"{feature_name} 的 X_train 与 y_train 样本数不一致")
        if x_test.shape[0] != y_test.shape[0]:
            raise ValueError(f"{feature_name} 的 X_test 与 y_test 样本数不一致")

        train_parts.append(x_train)
        test_parts.append(x_test)
        all_y_train.append(y_train)
        all_y_test.append(y_test)

    y_train, y_test = check_labels_consistent(all_y_train, all_y_test)

    X_train_merged = np.hstack(train_parts).astype(np.float32)
    X_test_merged = np.hstack(test_parts).astype(np.float32)

    merged_name = "_".join(FEATURES_TO_MERGE)

    print("\nMerged result")
    print("-" * 50)
    print(f"X_train merged shape: {X_train_merged.shape}")
    print(f"X_test  merged shape: {X_test_merged.shape}")
    print(f"y_train shape       : {y_train.shape}")
    print(f"y_test  shape       : {y_test.shape}")
    print(f"train labels        : pos={(y_train == 1).sum()}, neg={(y_train == 0).sum()}")
    print(f"test labels         : pos={(y_test == 1).sum()}, neg={(y_test == 0).sum()}")

    total_expected = sum(EXPECTED_DIMS.get(name, 0) for name in FEATURES_TO_MERGE if name in EXPECTED_DIMS)
    if total_expected > 0:
        print(f"expected total dim  : {total_expected}")

    # 保存 csv
    pd.DataFrame(X_train_merged).to_csv(MERGE_DIR / f"{merged_name}_X_train.csv", index=False)
    pd.DataFrame(X_test_merged).to_csv(MERGE_DIR / f"{merged_name}_X_test.csv", index=False)
    pd.DataFrame({"label": y_train}).to_csv(MERGE_DIR / f"{merged_name}_y_train.csv", index=False)
    pd.DataFrame({"label": y_test}).to_csv(MERGE_DIR / f"{merged_name}_y_test.csv", index=False)

    # 保存 numpy
    np.save(MERGE_DIR / f"{merged_name}_X_train.npy", X_train_merged)
    np.save(MERGE_DIR / f"{merged_name}_X_test.npy", X_test_merged)
    np.save(MERGE_DIR / f"{merged_name}_y_train.npy", y_train)
    np.save(MERGE_DIR / f"{merged_name}_y_test.npy", y_test)

    # 保存列来源说明
    col_ranges = []
    start = 0
    for feature_name, part in zip(FEATURES_TO_MERGE, train_parts):
        end_exclusive = start + part.shape[1]
        col_ranges.append({
            "feature_name": feature_name,
            "start_col": start,
            "end_col_inclusive": end_exclusive - 1,
            "end_col_exclusive": end_exclusive,
            "num_cols": part.shape[1],
        })
        start = end_exclusive

    pd.DataFrame(col_ranges).to_csv(MERGE_DIR / f"{merged_name}_column_ranges.csv", index=False)

    print("\nSaved files:")
    print(MERGE_DIR / f"{merged_name}_X_train.csv")
    print(MERGE_DIR / f"{merged_name}_X_test.csv")
    print(MERGE_DIR / f"{merged_name}_y_train.csv")
    print(MERGE_DIR / f"{merged_name}_y_test.csv")
    print(MERGE_DIR / f"{merged_name}_X_train.npy")
    print(MERGE_DIR / f"{merged_name}_X_test.npy")
    print(MERGE_DIR / f"{merged_name}_y_train.npy")
    print(MERGE_DIR / f"{merged_name}_y_test.npy")
    print(MERGE_DIR / f"{merged_name}_column_ranges.csv")


if __name__ == "__main__":
    main()