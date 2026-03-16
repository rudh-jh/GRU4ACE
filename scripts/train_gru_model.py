from pathlib import Path
import os
import json
import math
import random

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import tensorflow as tf
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc,
    average_precision_score,
    precision_recall_curve,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.layers import Dense, Dropout, Flatten, GRU
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical


# =========================
# 基础配置
# =========================
SEED = 42
N_SPLITS = 5
BATCH_SIZE = 28
EPOCHS = 50

MERGED_NAME = "BPF_FEGS_Fasttext_ProtT5_BERT_ESM2"
PREFIX = f"{MERGED_NAME}_EN"

ROOT = Path(r"E:\MYS\GRU4ACE")
BASE_DIR = ROOT / "outputs" / "split_seed42"
SEL_DIR = BASE_DIR / "selected" / "elastic_net" / MERGED_NAME
SAVE_DIR = BASE_DIR / "models" / "GRU_EN_306"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 随机种子
# =========================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def try_set_gpu_memory_growth():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass


# =========================
# 工具函数
# =========================
def categorical_probas_to_classes(p):
    return np.argmax(p, axis=1)


def calculate_performance(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    acc = float(tp + tn) / len(y_true)
    precision = float(tp) / (tp + fp + 1e-6)
    npv = float(tn) / (tn + fn + 1e-6)
    sensitivity = float(tp) / (tp + fn + 1e-6)
    specificity = float(tn) / (tn + fp + 1e-6)
    mcc = float(tp * tn - fp * fn) / (
        math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-6
    )
    f1 = float(tp * 2) / (tp * 2 + fp + fn + 1e-6)

    return {
        "acc": acc,
        "precision": precision,
        "npv": npv,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "mcc": mcc,
        "f1": f1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def load_selected_data():
    X_train = np.load(SEL_DIR / f"{PREFIX}_X_train.npy")
    X_test = np.load(SEL_DIR / f"{PREFIX}_X_test.npy")
    y_train = np.load(SEL_DIR / f"{PREFIX}_y_train.npy").reshape(-1)
    y_test = np.load(SEL_DIR / f"{PREFIX}_y_test.npy").reshape(-1)

    print(f"[INFO] X_train: {X_train.shape}")
    print(f"[INFO] X_test : {X_test.shape}")
    print(f"[INFO] y_train: {y_train.shape}")
    print(f"[INFO] y_test : {y_test.shape}")

    return X_train.astype(np.float32), X_test.astype(np.float32), y_train.astype(int), y_test.astype(int)


def build_gru_model(input_dim: int):
    model = Sequential()
    model.add(tf.keras.Input(shape=(1, input_dim)))
    model.add(GRU(32, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(GRU(16, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(64, activation="sigmoid", name="Dense_64"))
    model.add(Dropout(0.7))
    model.add(Dense(16, activation="sigmoid", name="Dense_16"))
    model.add(Dense(2, activation="sigmoid", name="Dense_2"))
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adamax",
        metrics=["accuracy"],
    )
    return model


def save_history_plot(history, save_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training History")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

# =========================
# 主流程
# =========================
def main():
    seed_everything(SEED)
    try_set_gpu_memory_growth()

    X_train_raw, X_test_raw, y_train_raw, y_test = load_selected_data()

    # 只对训练集做欠采样，和你 notebook 当前活跃逻辑保持一致
    rus = RandomUnderSampler(random_state=SEED)
    X_train_bal, y_train_bal = rus.fit_resample(X_train_raw, y_train_raw)

    print(f"[INFO] balanced train shape: {X_train_bal.shape}")
    print(f"[INFO] balanced train labels: pos={(y_train_bal == 1).sum()}, neg={(y_train_bal == 0).sum()}")

    # 保存 notebook 风格的中间数据
    train_bal_df = pd.DataFrame(X_train_bal)
    train_bal_df["label"] = y_train_bal
    train_bal_df.to_csv(SAVE_DIR / "XtrainData_balanced.csv", index=False)

    test_df = pd.DataFrame(X_test_raw)
    test_df["label"] = y_test
    test_df.to_csv(SAVE_DIR / "XtestData.csv", index=False)

    input_dim = X_train_bal.shape[1]
    X_bal_3d = np.reshape(X_train_bal, (-1, 1, input_dim))
    X_test_3d = np.reshape(X_test_raw, (-1, 1, input_dim))

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    cv_records = []
    cv_ytest_stack = []
    cv_yscore_stack = []

    print("\n[INFO] Start 5-fold cross validation")
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_bal_3d, y_train_bal), start=1):
        print(f"\n========== Fold {fold_idx}/{N_SPLITS} ==========")

        X_tr = X_bal_3d[tr_idx]
        X_va = X_bal_3d[va_idx]
        y_tr = y_train_bal[tr_idx]
        y_va = y_train_bal[va_idx]

        y_tr_cat = to_categorical(y_tr, num_classes=2)
        y_va_cat = to_categorical(y_va, num_classes=2)

        model = build_gru_model(input_dim)

        model_path = SAVE_DIR / f"{fold_idx}GRU_new.keras"
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
            ModelCheckpoint(str(model_path), monitor="val_loss", save_best_only=True, verbose=1),
        ]

        history = model.fit(
            X_tr,
            y_tr_cat,
            validation_data=(X_va, y_va_cat),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=1,
            callbacks=callbacks,
        )

        save_history_plot(history, SAVE_DIR / f"fold_{fold_idx}_history.png")

        # 读取最佳模型
        best_model = load_model(model_path)

        y_score = best_model.predict(X_va, verbose=0)
        y_pred = categorical_probas_to_classes(y_score)

        perf = calculate_performance(y_va, y_pred)
        fpr, tpr, _ = roc_curve(y_va, y_score[:, 1])
        roc_auc = auc(fpr, tpr)
        aupr = average_precision_score(y_va, y_score[:, 1])

        record = {
            "fold": fold_idx,
            "acc": perf["acc"],
            "precision": perf["precision"],
            "npv": perf["npv"],
            "sensitivity": perf["sensitivity"],
            "specificity": perf["specificity"],
            "mcc": perf["mcc"],
            "f1": perf["f1"],
            "roc_auc": roc_auc,
            "aupr": aupr,
            "tn": perf["tn"],
            "fp": perf["fp"],
            "fn": perf["fn"],
            "tp": perf["tp"],
        }
        cv_records.append(record)

        cv_ytest_stack.append(y_va_cat)
        cv_yscore_stack.append(y_score)

        print(
            "[CV] "
            f"acc={record['acc']:.6f}, "
            f"Sn={record['sensitivity']:.6f}, "
            f"Sp={record['specificity']:.6f}, "
            f"MCC={record['mcc']:.6f}, "
            f"F1={record['f1']:.6f}, "
            f"AUC={record['roc_auc']:.6f}, "
            f"AUPR={record['aupr']:.6f}"
        )

    cv_df = pd.DataFrame(cv_records)
    cv_mean = cv_df.mean(numeric_only=True)
    cv_mean_row = {"fold": "mean"}
    for col in cv_df.columns:
        if col != "fold":
            cv_mean_row[col] = cv_mean[col]
    cv_df = pd.concat([cv_df, pd.DataFrame([cv_mean_row])], ignore_index=True)

    cv_df.to_csv(SAVE_DIR / "GRU_CV_results.csv", index=False)

    cv_ytest_all = np.vstack(cv_ytest_stack)
    cv_yscore_all = np.vstack(cv_yscore_stack)
    pd.DataFrame(cv_ytest_all, columns=["Class_0", "Class_1"]).to_csv(SAVE_DIR / "GRU_ytest_cv.csv", index=False)
    pd.DataFrame(cv_yscore_all, columns=["Score_0", "Score_1"]).to_csv(SAVE_DIR / "GRU_yscore_cv.csv", index=False)

    print("\n[INFO] Cross-validation summary")
    print(cv_df)

    # =========================
    # 固定 test 集评估
    # =========================
    print("\n[INFO] Evaluate on fixed test set")
    test_records = []
    test_ytest_stack = []
    test_yscore_stack = []

    for fold_idx in range(1, N_SPLITS + 1):
        model_path = SAVE_DIR / f"{fold_idx}GRU_new.keras"
        model = load_model(model_path)

        y_score = model.predict(X_test_3d, verbose=0)
        y_pred = categorical_probas_to_classes(y_score)

        perf = calculate_performance(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
        roc_auc = auc(fpr, tpr)
        aupr = average_precision_score(y_test, y_score[:, 1])

        record = {
            "fold": fold_idx,
            "acc": perf["acc"],
            "precision": perf["precision"],
            "npv": perf["npv"],
            "sensitivity": perf["sensitivity"],
            "specificity": perf["specificity"],
            "mcc": perf["mcc"],
            "f1": perf["f1"],
            "roc_auc": roc_auc,
            "aupr": aupr,
            "tn": perf["tn"],
            "fp": perf["fp"],
            "fn": perf["fn"],
            "tp": perf["tp"],
        }
        test_records.append(record)

        test_ytest_stack.append(to_categorical(y_test, num_classes=2))
        test_yscore_stack.append(y_score)

        print(
            "[TEST] "
            f"fold={fold_idx}, "
            f"acc={record['acc']:.6f}, "
            f"Sn={record['sensitivity']:.6f}, "
            f"Sp={record['specificity']:.6f}, "
            f"MCC={record['mcc']:.6f}, "
            f"F1={record['f1']:.6f}, "
            f"AUC={record['roc_auc']:.6f}, "
            f"AUPR={record['aupr']:.6f}"
        )

    test_df = pd.DataFrame(test_records)
    test_mean = test_df.mean(numeric_only=True)
    test_mean_row = {"fold": "mean"}
    for col in test_df.columns:
        if col != "fold":
            test_mean_row[col] = test_mean[col]
    test_df = pd.concat([test_df, pd.DataFrame([test_mean_row])], ignore_index=True)

    test_df.to_csv(SAVE_DIR / "GRU_test_results.csv", index=False)

    test_ytest_all = np.vstack(test_ytest_stack)
    test_yscore_all = np.vstack(test_yscore_stack)
    pd.DataFrame(test_ytest_all, columns=["Class_0", "Class_1"]).to_csv(SAVE_DIR / "GRU_ytest_test.csv", index=False)
    pd.DataFrame(test_yscore_all, columns=["Score_0", "Score_1"]).to_csv(SAVE_DIR / "GRU_yscore_test.csv", index=False)

    print("\n[INFO] Test summary")
    print(test_df)

    summary = {
        "seed": SEED,
        "n_splits": N_SPLITS,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "train_raw_shape": list(X_train_raw.shape),
        "test_shape": list(X_test_raw.shape),
        "train_balanced_shape": list(X_train_bal.shape),
        "model_dir": str(SAVE_DIR),
        "input_dim": int(input_dim),
    }
    with open(SAVE_DIR / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] saved to: {SAVE_DIR}")


if __name__ == "__main__":
    main()