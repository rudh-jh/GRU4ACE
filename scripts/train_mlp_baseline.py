from pathlib import Path
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "outputs" / "split_seed42"
SEL_DIR = BASE_DIR / "selected" / "elastic_net"
SAVE_DIR = BASE_DIR / "baseline" / "BPF_FEGS_EN_MLP"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def load_selected_data():
    X_train = np.load(SEL_DIR / "BPF_FEGS_EN_X_train.npy")
    X_test = np.load(SEL_DIR / "BPF_FEGS_EN_X_test.npy")
    y_train = np.load(SEL_DIR / "BPF_FEGS_EN_y_train.npy")
    y_test = np.load(SEL_DIR / "BPF_FEGS_EN_y_test.npy")

    print(f"[INFO] X_train: {X_train.shape}")
    print(f"[INFO] X_test : {X_test.shape}")
    print(f"[INFO] y_train: {y_train.shape}")
    print(f"[INFO] y_test : {y_test.shape}")

    return X_train, X_test, y_train, y_test


def calc_metrics(y_true, y_pred, y_score):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    acc = accuracy_score(y_true, y_pred)
    sn = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    mcc = matthews_corrcoef(y_true, y_pred)

    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = None

    return {
        "ACC": round(acc, 6),
        "Sn": round(sn, 6),
        "Sp": round(sp, 6),
        "MCC": round(mcc, 6),
        "AUC": None if auc is None else round(auc, 6),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def main():
    X_train, X_test, y_train, y_test = load_selected_data()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        batch_size=32,
        learning_rate_init=1e-3,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
    )

    clf.fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)
    y_train_score = clf.predict_proba(X_train)[:, 1]

    y_test_pred = clf.predict(X_test)
    y_test_score = clf.predict_proba(X_test)[:, 1]

    train_metrics = calc_metrics(y_train, y_train_pred, y_train_score)
    test_metrics = calc_metrics(y_test, y_test_pred, y_test_score)

    print("\n[TRAIN METRICS]")
    for k, v in train_metrics.items():
        print(f"{k}: {v}")

    print("\n[TEST METRICS]")
    for k, v in test_metrics.items():
        print(f"{k}: {v}")

    np.save(SAVE_DIR / "scaler_mean.npy", scaler.mean_)
    np.save(SAVE_DIR / "scaler_scale.npy", scaler.scale_)

    with open(SAVE_DIR / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(train_metrics, f, ensure_ascii=False, indent=2)

    with open(SAVE_DIR / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)

    np.savetxt(
        SAVE_DIR / "train_predictions.csv",
        np.column_stack([y_train, y_train_pred, y_train_score]),
        delimiter=",",
        header="y_true,y_pred,y_score",
        comments="",
    )

    np.savetxt(
        SAVE_DIR / "test_predictions.csv",
        np.column_stack([y_test, y_test_pred, y_test_score]),
        delimiter=",",
        header="y_true,y_pred,y_score",
        comments="",
    )

    summary = {
        "input_dim": int(X_train.shape[1]),
        "hidden_layer_sizes": [128, 64],
        "alpha": 1e-3,
        "learning_rate_init": 1e-3,
        "max_iter": 300,
        "random_state": 42,
    }
    with open(SAVE_DIR / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] results saved to: {SAVE_DIR}")


if __name__ == "__main__":
    main()