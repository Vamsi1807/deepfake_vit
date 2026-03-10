import os
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, LSTM, Masking
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "extracted_features_vit"
MAX_LEN = 30


def find_feature_roots(root):
    roots = []
    if not os.path.isdir(root):
        return roots

    legacy_real = os.path.join(root, "real")
    legacy_fake = os.path.join(root, "fake")
    if os.path.isdir(legacy_real) and os.path.isdir(legacy_fake):
        roots.append(root)

    for name in os.listdir(root):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        if os.path.isdir(os.path.join(path, "real")) and os.path.isdir(os.path.join(path, "fake")):
            roots.append(path)

    return roots


def load_features(root):
    X_local = []
    y_local = []
    for label in ["real", "fake"]:
        label_dir = os.path.join(root, label)
        if not os.path.isdir(label_dir):
            continue
        for file in os.listdir(label_dir):
            if not file.endswith(".npy"):
                continue
            path = os.path.join(label_dir, file)
            sequence = np.load(path)
            if sequence.size == 0:
                continue
            X_local.append(sequence)
            y_local.append(0 if label == "real" else 1)
    return X_local, y_local


X = []
y = []
feature_roots = find_feature_roots(str(DATA_DIR))
if not feature_roots:
    raise FileNotFoundError(
        f"No extracted features found in {DATA_DIR}. Run feature_extraction.py first."
    )

for root in feature_roots:
    X_part, y_part = load_features(root)
    X.extend(X_part)
    y.extend(y_part)
    print(f"Loaded {len(y_part)} samples from {root}")

if not X:
    raise RuntimeError("No valid sequences were loaded from extracted features.")

feature_dim = X[0].shape[1]
print(f"Detected feature dimension: {feature_dim}")

X_pad = pad_sequences(X, maxlen=MAX_LEN, dtype="float32", padding="post")
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_pad, y, test_size=0.2, random_state=42, stratify=y
)

model = Sequential(
    [
        Masking(mask_value=0.0, input_shape=(MAX_LEN, feature_dim)),
        LSTM(128),
        Dropout(0.5),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ]
)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

models_dir = BASE_DIR / "models"
os.makedirs(models_dir, exist_ok=True)
checkpoint = ModelCheckpoint(
    str(models_dir / "best_lstm_vit.h5"),
    save_best_only=True,
    monitor="val_accuracy",
    mode="max",
)

model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=16,
    validation_split=0.2,
    callbacks=[checkpoint],
)

best_model_path = str(models_dir / "best_lstm_vit.h5")
best_model = load_model(best_model_path)

loss, acc = best_model.evaluate(X_test, y_test, verbose=0)
print(f"Best-model Test Loss: {loss:.4f}")
print(f"Best-model Test Accuracy: {acc:.4f}")

y_prob = best_model.predict(X_test, verbose=0).ravel()
y_pred = (y_prob >= 0.5).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

metrics = {
    "test_loss": float(loss),
    "test_accuracy": float(accuracy_score(y_test, y_pred)),
    "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred, zero_division=0)),
    "recall_sensitivity": float(recall_score(y_test, y_pred, zero_division=0)),
    "specificity": float(specificity),
    "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, y_prob)),
    "pr_auc": float(average_precision_score(y_test, y_prob)),
    "mcc": float(matthews_corrcoef(y_test, y_pred)),
    "cohen_kappa": float(cohen_kappa_score(y_test, y_pred)),
    "brier_score": float(brier_score_loss(y_test, y_prob)),
    "confusion_matrix": {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    },
}

# Bootstrap confidence intervals as a validity check on holdout performance.
rng = np.random.default_rng(42)
n_bootstrap = 500
acc_samples = []
auc_samples = []
indices = np.arange(len(y_test))

for _ in range(n_bootstrap):
    sample_idx = rng.choice(indices, size=len(indices), replace=True)
    y_t = y_test[sample_idx]
    y_p = y_pred[sample_idx]
    y_s = y_prob[sample_idx]
    acc_samples.append(accuracy_score(y_t, y_p))
    if len(np.unique(y_t)) == 2:
        auc_samples.append(roc_auc_score(y_t, y_s))

metrics["validity"] = {
    "bootstrap_samples": n_bootstrap,
    "accuracy_ci95": [
        float(np.percentile(acc_samples, 2.5)),
        float(np.percentile(acc_samples, 97.5)),
    ],
    "roc_auc_ci95": (
        [
            float(np.percentile(auc_samples, 2.5)),
            float(np.percentile(auc_samples, 97.5)),
        ]
        if auc_samples
        else None
    ),
}

report_text = classification_report(y_test, y_pred, target_names=["real", "fake"], zero_division=0)
metrics["classification_report"] = report_text

print("\n=== Classification Report ===")
print(report_text)
print("=== Confusion Matrix ===")
print(np.array([[tn, fp], [fn, tp]]))
print("=== Core Metrics ===")
for key in [
    "test_accuracy",
    "balanced_accuracy",
    "precision",
    "recall_sensitivity",
    "specificity",
    "f1_score",
    "roc_auc",
    "pr_auc",
    "mcc",
    "cohen_kappa",
]:
    print(f"{key}: {metrics[key]:.4f}")
print(
    "95% CI (accuracy): "
    f"{metrics['validity']['accuracy_ci95'][0]:.4f} to {metrics['validity']['accuracy_ci95'][1]:.4f}"
)
if metrics["validity"]["roc_auc_ci95"] is not None:
    print(
        "95% CI (roc_auc): "
        f"{metrics['validity']['roc_auc_ci95'][0]:.4f} to {metrics['validity']['roc_auc_ci95'][1]:.4f}"
    )

with open(models_dir / "metrics_vit.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

with open(models_dir / "classification_report_vit.txt", "w", encoding="utf-8") as f:
    f.write(report_text)

print(f"\nSaved metrics to: {models_dir / 'metrics_vit.json'}")
print(f"Saved classification report to: {models_dir / 'classification_report_vit.txt'}")
