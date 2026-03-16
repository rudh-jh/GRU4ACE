from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
)
from imblearn.over_sampling import SMOTE


ROOT = Path(r"E:\MYS\GRU4ACE")
OUT_DIR = ROOT / "outputs"
SPLIT_DIR = OUT_DIR / "split_seed42"
MERGE_DIR = SPLIT_DIR / "merged"

# 这里对应你 merge_features.py 生成的文件前缀
MERGED_NAME = "BPF_FEGS"

BASELINE_DIR = SPLIT_DIR / "baseline" / MERGED_NAME
BASELINE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42


def load_data():
    x_train_file = MERGE_DIR / f"{MERGED_NAME}_X_train.csv"
    x_test_file = MERGE_DIR / f"{MERGED_NAME}_X_test.csv"
    y_train_file = MERGE_DIR / f"{MERGED_NAME}_y_train.csv"
    y_test_file = MERGE_DIR / f"{MERGED_NAME}_y_test.csv"

    if not x_train_file.exists():
        raise FileNotFoundError(f"找不到文件: {x_train_file}")
    if not x_test_file.exists():
        raise FileNotFoundError(f"找不到文件: {x_test_file}")
    if not y_train_file.exists():
        raise FileNotFoundError(f"找不到文件: {y_train_file}")
    if not y_test_file.exists():
        raise FileNotFoundError(f"找不到文件: {y_test_file}")

    X_train = pd.read_csv(x_train_file).to_numpy(dtype=np.float32)
    X_test = pd.read_csv(x_test_file).to_numpy(dtype=np.float32)
    y_train = pd.read_csv(y_train_file)["label"].to_numpy(dtype=np.int32)
    y_test = pd.read_csv(y_test_file)["label"].to_numpy(dtype=np.int32)

    return X_train, X_test, y_train, y_test


def calculate_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sn = recall_score(y_true, y_pred)  # sensitivity / recall of positive class
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)

    metrics = {
        "ACC": float(acc),
        "BACC": float(bacc),
        "SN": float(sn),
        "SP": float(sp),
        "F1": float(f1),
        "MCC": float(mcc),
        "AUC": float(auc),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }
    return metrics


def main():
    print(f"Loading merged feature set: {MERGED_NAME}")
    X_train, X_test, y_train, y_test = load_data()

    print("Original shapes")
    print(f"X_train: {X_train.shape}")
    print(f"X_test : {X_test.shape}")
    print(f"y_train: {y_train.shape}, pos={(y_train == 1).sum()}, neg={(y_train == 0).sum()}")
    print(f"y_test : {y_test.shape}, pos={(y_test == 1).sum()}, neg={(y_test == 0).sum()}")

    # 1) 只在训练集上拟合标准化器
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 2) 只在训练集上做 SMOTE
    smote = SMOTE(random_state=SEED)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

    print("\nAfter SMOTE")
    print(f"X_train_smote: {X_train_smote.shape}")
    print(f"y_train_smote: {y_train_smote.shape}, pos={(y_train_smote == 1).sum()}, neg={(y_train_smote == 0).sum()}")

    # 3) baseline 模型：Logistic Regression
    model = LogisticRegression(
        random_state=SEED,
        max_iter=2000,
        solver="lbfgs",
    )
    model.fit(X_train_smote, y_train_smote)

    # 4) 训练集评估（在 SMOTE 后训练集上）
    train_pred = model.predict(X_train_smote)
    train_prob = model.predict_proba(X_train_smote)[:, 1]
    train_metrics = calculate_metrics(y_train_smote, train_pred, train_prob)

    # 5) 独立测试集评估（不做 SMOTE）
    test_pred = model.predict(X_test_scaled)
    test_prob = model.predict_proba(X_test_scaled)[:, 1]
    test_metrics = calculate_metrics(y_test, test_pred, test_prob)

    print("\nTrain metrics")
    for k, v in train_metrics.items():
        print(f"{k}: {v}")

    print("\nTest metrics")
    for k, v in test_metrics.items():
        print(f"{k}: {v}")

    # 6) 保存模型和 scaler
    joblib.dump(scaler, BASELINE_DIR / "scaler.joblib")
    joblib.dump(model, BASELINE_DIR / "logistic_regression.joblib")

    # 7) 保存指标
    with open(BASELINE_DIR / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(train_metrics, f, ensure_ascii=False, indent=2)

    with open(BASELINE_DIR / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)

    # 8) 保存测试集预测结果
    test_pred_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred": test_pred,
        "y_prob": test_prob,
    })
    test_pred_df.to_csv(BASELINE_DIR / "test_predictions.csv", index=False)

    # 9) 保存训练集预测结果
    train_pred_df = pd.DataFrame({
        "y_true": y_train_smote,
        "y_pred": train_pred,
        "y_prob": train_prob,
    })
    train_pred_df.to_csv(BASELINE_DIR / "train_predictions.csv", index=False)

    # 10) 保存运行摘要
    run_summary = {
        "merged_name": MERGED_NAME,
        "seed": SEED,
        "train_shape_before_smote": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "train_shape_after_smote": list(X_train_smote.shape),
        "train_positive_before_smote": int((y_train == 1).sum()),
        "train_negative_before_smote": int((y_train == 0).sum()),
        "train_positive_after_smote": int((y_train_smote == 1).sum()),
        "train_negative_after_smote": int((y_train_smote == 0).sum()),
        "test_positive": int((y_test == 1).sum()),
        "test_negative": int((y_test == 0).sum()),
        "model": "LogisticRegression",
    }

    with open(BASELINE_DIR / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)

    print("\nSaved files:")
    print(BASELINE_DIR / "scaler.joblib")
    print(BASELINE_DIR / "logistic_regression.joblib")
    print(BASELINE_DIR / "train_metrics.json")
    print(BASELINE_DIR / "test_metrics.json")
    print(BASELINE_DIR / "train_predictions.csv")
    print(BASELINE_DIR / "test_predictions.csv")
    print(BASELINE_DIR / "run_summary.json")


if __name__ == "__main__":
    main()