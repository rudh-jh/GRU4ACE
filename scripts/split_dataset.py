from pathlib import Path
import json
import numpy as np
import pandas as pd


ROOT = Path(r"E:\MYS\GRU4ACE")
OUT_DIR = ROOT / "outputs"
SPLIT_DIR = OUT_DIR / "split_seed42"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LABEL_FILE = OUT_DIR / "seq_label_all.csv"

SEED = 42

# 按论文里的数量固定切分
TEST_POS = 76
TEST_NEG = 128


def main():
    if not SEQ_LABEL_FILE.exists():
        raise FileNotFoundError(f"找不到文件: {SEQ_LABEL_FILE}")

    df = pd.read_csv(SEQ_LABEL_FILE)

    if "sequence" not in df.columns or "label" not in df.columns:
        raise ValueError("seq_label_all.csv 必须包含 'sequence' 和 'label' 两列")

    df = df.copy()
    df["sample_id"] = np.arange(len(df))

    pos_idx = df.index[df["label"] == 1].to_numpy()
    neg_idx = df.index[df["label"] == 0].to_numpy()

    print(f"total samples   : {len(df)}")
    print(f"positive samples: {len(pos_idx)}")
    print(f"negative samples: {len(neg_idx)}")

    if len(pos_idx) != 394 or len(neg_idx) != 626:
        print("[WARN] 当前样本数量与论文 394/626 不一致，请先确认数据。")

    if TEST_POS > len(pos_idx) or TEST_NEG > len(neg_idx):
        raise ValueError("测试集数量超过现有样本数")

    rng = np.random.default_rng(SEED)

    pos_idx_shuffled = pos_idx.copy()
    neg_idx_shuffled = neg_idx.copy()
    rng.shuffle(pos_idx_shuffled)
    rng.shuffle(neg_idx_shuffled)

    test_pos_idx = pos_idx_shuffled[:TEST_POS]
    train_pos_idx = pos_idx_shuffled[TEST_POS:]

    test_neg_idx = neg_idx_shuffled[:TEST_NEG]
    train_neg_idx = neg_idx_shuffled[TEST_NEG:]

    train_idx = np.concatenate([train_pos_idx, train_neg_idx])
    test_idx = np.concatenate([test_pos_idx, test_neg_idx])

    # 再打乱一次，避免全是正样本在前、负样本在后
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    train_df = df.loc[train_idx].reset_index(drop=True)
    test_df = df.loc[test_idx].reset_index(drop=True)

    print("\nSplit summary")
    print("-" * 40)
    print(f"train size    : {len(train_df)}")
    print(f"test size     : {len(test_df)}")
    print(f"train positive: {(train_df['label'] == 1).sum()}")
    print(f"train negative: {(train_df['label'] == 0).sum()}")
    print(f"test positive : {(test_df['label'] == 1).sum()}")
    print(f"test negative : {(test_df['label'] == 0).sum()}")

    # 保存索引，后面切各种特征矩阵时直接复用
    np.save(SPLIT_DIR / "train_idx.npy", train_idx)
    np.save(SPLIT_DIR / "test_idx.npy", test_idx)

    # 保存标签，方便核对
    np.save(SPLIT_DIR / "y_train.npy", train_df["label"].to_numpy())
    np.save(SPLIT_DIR / "y_test.npy", test_df["label"].to_numpy())

    # 保存样本表
    train_df.to_csv(SPLIT_DIR / "train_seq_label.csv", index=False)
    test_df.to_csv(SPLIT_DIR / "test_seq_label.csv", index=False)

    # 保存纯 sample_id 形式
    pd.DataFrame({"sample_id": train_idx}).to_csv(SPLIT_DIR / "train_ids.csv", index=False)
    pd.DataFrame({"sample_id": test_idx}).to_csv(SPLIT_DIR / "test_ids.csv", index=False)

    summary = {
        "seed": SEED,
        "total_samples": int(len(df)),
        "total_positive": int((df["label"] == 1).sum()),
        "total_negative": int((df["label"] == 0).sum()),
        "test_positive_fixed": TEST_POS,
        "test_negative_fixed": TEST_NEG,
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "train_positive": int((train_df["label"] == 1).sum()),
        "train_negative": int((train_df["label"] == 0).sum()),
        "test_positive": int((test_df["label"] == 1).sum()),
        "test_negative": int((test_df["label"] == 0).sum()),
    }

    with open(SPLIT_DIR / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nSaved files:")
    print(SPLIT_DIR / "train_idx.npy")
    print(SPLIT_DIR / "test_idx.npy")
    print(SPLIT_DIR / "y_train.npy")
    print(SPLIT_DIR / "y_test.npy")
    print(SPLIT_DIR / "train_seq_label.csv")
    print(SPLIT_DIR / "test_seq_label.csv")
    print(SPLIT_DIR / "train_ids.csv")
    print(SPLIT_DIR / "test_ids.csv")
    print(SPLIT_DIR / "split_summary.json")


if __name__ == "__main__":
    main()