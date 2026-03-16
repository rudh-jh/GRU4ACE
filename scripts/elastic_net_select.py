from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "outputs" / "split_seed42"
MERGED_DIR = BASE_DIR / "merged"

MERGED_NAME = "BPF_FEGS_Fasttext_ProtT5_BERT_ESM2"

SAVE_DIR = BASE_DIR / "selected" / "elastic_net" / MERGED_NAME
SAVE_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_SPECS = [
    ("BPF", 441),
    ("FEGS", 578),
    ("Fasttext", 120),
    ("ProtT5", 1024),
    ("BERT", 768),
    ("ESM2", 320),
]


def load_data():
    X_train = np.load(MERGED_DIR / f"{MERGED_NAME}_X_train.npy")
    X_test = np.load(MERGED_DIR / f"{MERGED_NAME}_X_test.npy")
    y_train = np.load(MERGED_DIR / f"{MERGED_NAME}_y_train.npy")
    y_test = np.load(MERGED_DIR / f"{MERGED_NAME}_y_test.npy")

    print(f"[INFO] X_train: {X_train.shape}")
    print(f"[INFO] X_test : {X_test.shape}")
    print(f"[INFO] y_train: {y_train.shape}")
    print(f"[INFO] y_test : {y_test.shape}")

    return X_train, X_test, y_train, y_test


def build_feature_names():
    names = []
    for feature_name, dim in FEATURE_SPECS:
        names.extend([f"{feature_name}_{i}" for i in range(dim)])
    return np.array(names, dtype=object)


def build_group_ranges():
    ranges = []
    start = 0
    for feature_name, dim in FEATURE_SPECS:
        end = start + dim
        ranges.append({
            "feature_name": feature_name,
            "start_col": start,
            "end_col_inclusive": end - 1,
            "end_col_exclusive": end,
            "num_cols": dim,
        })
        start = end
    return pd.DataFrame(ranges)


def summarize_selected_groups(selected_idx):
    group_rows = []
    start = 0
    for feature_name, dim in FEATURE_SPECS:
        end = start + dim
        mask = (selected_idx >= start) & (selected_idx < end)
        count = int(mask.sum())
        group_rows.append({
            "feature_name": feature_name,
            "start_col": start,
            "end_col_exclusive": end,
            "original_dim": dim,
            "selected_dim": count,
            "selected_ratio": round(count / dim, 6),
        })
        start = end
    return pd.DataFrame(group_rows)


def main():
    X_train, X_test, y_train, y_test = load_data()
    feature_names = build_feature_names()

    if X_train.shape[1] != len(feature_names):
        raise ValueError(
            f"特征维度不一致: X_train={X_train.shape[1]}, feature_names={len(feature_names)}"
        )

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    selector = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
        C=0.1,
        max_iter=5000,
        random_state=42,
        n_jobs=-1,
    )

    selector.fit(X_train_std, y_train)

    coef = selector.coef_.ravel()
    selected_mask = coef != 0
    selected_idx = np.where(selected_mask)[0]
    selected_coef = coef[selected_mask]
    selected_names = feature_names[selected_mask]

    num_selected = len(selected_idx)
    print(f"[INFO] total features   : {X_train.shape[1]}")
    print(f"[INFO] selected features: {num_selected}")

    if num_selected == 0:
        raise ValueError(
            "Elastic Net 没选出任何特征。把 C 调大一些，例如 0.5 / 1.0，或者把 l1_ratio 调小。"
        )

    X_train_sel = X_train[:, selected_idx]
    X_test_sel = X_test[:, selected_idx]

    print(f"[INFO] X_train_selected: {X_train_sel.shape}")
    print(f"[INFO] X_test_selected : {X_test_sel.shape}")

    prefix = f"{MERGED_NAME}_EN"

    # 保存选中特征矩阵
    np.save(SAVE_DIR / f"{prefix}_X_train.npy", X_train_sel)
    np.save(SAVE_DIR / f"{prefix}_X_test.npy", X_test_sel)
    np.save(SAVE_DIR / f"{prefix}_y_train.npy", y_train)
    np.save(SAVE_DIR / f"{prefix}_y_test.npy", y_test)

    pd.DataFrame(X_train_sel).to_csv(SAVE_DIR / f"{prefix}_X_train.csv", index=False)
    pd.DataFrame(X_test_sel).to_csv(SAVE_DIR / f"{prefix}_X_test.csv", index=False)
    pd.DataFrame({"label": y_train}).to_csv(SAVE_DIR / f"{prefix}_y_train.csv", index=False)
    pd.DataFrame({"label": y_test}).to_csv(SAVE_DIR / f"{prefix}_y_test.csv", index=False)

    # 保存 scaler，后面训练模型时可复用
    np.save(SAVE_DIR / "scaler_mean.npy", scaler.mean_)
    np.save(SAVE_DIR / "scaler_scale.npy", scaler.scale_)

    # 保存所有原始列分组信息
    build_group_ranges().to_csv(SAVE_DIR / "all_feature_ranges.csv", index=False)

    # 保存选中特征详情
    selected_df = pd.DataFrame({
        "selected_index": selected_idx,
        "feature_name": selected_names,
        "coefficient": selected_coef,
        "abs_coefficient": np.abs(selected_coef),
    }).sort_values("abs_coefficient", ascending=False)

    selected_df.to_csv(SAVE_DIR / "selected_features.csv", index=False)

    # 保存各特征组被选中的数量
    selected_group_df = summarize_selected_groups(selected_idx)
    selected_group_df.to_csv(SAVE_DIR / "selected_group_summary.csv", index=False)

    summary = {
        "merged_name": MERGED_NAME,
        "input_dim": int(X_train.shape[1]),
        "selected_dim": int(num_selected),
        "l1_ratio": 0.5,
        "C": 0.1,
        "max_iter": 5000,
        "random_state": 42,
        "train_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0]),
    }

    with open(SAVE_DIR / "selection_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] saved to: {SAVE_DIR}")


if __name__ == "__main__":
    main()