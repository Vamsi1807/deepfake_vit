import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, LSTM, Masking
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "extracted_features_vit"
MAX_LEN = 30
TEST_SIZE = 0.2
VAL_SIZE = 0.2
TARGET_SPECIFICITY = 0.97
RANDOM_STATE = 42


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


def parse_source_video_id(root, filename):
    stem = Path(filename).stem
    if "_face_" in stem:
        source_stem = stem.rsplit("_face_", 1)[0]
    else:
        source_stem = stem
    return f"{Path(root).name}:{source_stem}"


def load_feature_samples(root):
    samples = []
    for label_name, label_value in (("real", 0), ("fake", 1)):
        label_dir = os.path.join(root, label_name)
        if not os.path.isdir(label_dir):
            continue
        for file in os.listdir(label_dir):
            if not file.endswith(".npy"):
                continue
            path = os.path.join(label_dir, file)
            sequence = np.load(path)
            if sequence.size == 0:
                continue
            samples.append(
                {
                    "sequence": sequence,
                    "label": label_value,
                    "source_video_id": parse_source_video_id(root, file),
                    "feature_path": path,
                }
            )
    return samples


def build_group_records(samples):
    grouped = {}
    for sample in samples:
        source_video_id = sample["source_video_id"]
        grouped.setdefault(source_video_id, sample["label"])
    group_ids = np.array(list(grouped.keys()))
    group_labels = np.array([grouped[group_id] for group_id in group_ids], dtype=np.int32)
    return group_ids, group_labels


def stratified_group_split(group_ids, group_labels, test_size, random_state):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(group_ids, group_labels))
    return group_ids[train_idx], group_ids[test_idx]


def select_samples_by_groups(samples, allowed_groups):
    allowed_groups = set(allowed_groups)
    return [sample for sample in samples if sample["source_video_id"] in allowed_groups]


def prepare_xy(samples):
    sequences = [sample["sequence"] for sample in samples]
    labels = np.array([sample["label"] for sample in samples], dtype=np.int32)
    x_pad = pad_sequences(sequences, maxlen=MAX_LEN, dtype="float32", padding="post")
    return x_pad, labels


def build_model(feature_dim):
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
    return model


def metrics_at_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    return {
        "threshold": float(threshold),
        "y_pred": y_pred,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "specificity": float(specificity),
        "sensitivity": float(sensitivity),
        "balanced_accuracy": float(balanced_acc),
    }


def choose_threshold(y_true, y_prob, target_specificity):
    candidate_thresholds = np.unique(np.concatenate(([0.0, 0.5, 1.0], y_prob)))
    evaluations = [metrics_at_threshold(y_true, y_prob, threshold) for threshold in candidate_thresholds]
    eligible = [entry for entry in evaluations if entry["specificity"] >= target_specificity]
    pool = eligible if eligible else evaluations
    best = max(
        pool,
        key=lambda entry: (
            entry["balanced_accuracy"],
            entry["specificity"],
            -entry["threshold"],
        ),
    )
    return best


samples = []
feature_roots = find_feature_roots(str(DATA_DIR))
if not feature_roots:
    raise FileNotFoundError(
        f"No extracted features found in {DATA_DIR}. Run feature_extraction.py first."
    )

for root in feature_roots:
    root_samples = load_feature_samples(root)
    samples.extend(root_samples)
    print(f"Loaded {len(root_samples)} samples from {root}")

if not samples:
    raise RuntimeError("No valid sequences were loaded from extracted features.")

feature_dim = samples[0]["sequence"].shape[1]
print(f"Detected feature dimension: {feature_dim}")

group_ids, group_labels = build_group_records(samples)
print(f"Unique source videos detected: {len(group_ids)}")

train_val_groups, test_groups = stratified_group_split(
    group_ids, group_labels, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
train_val_group_labels = np.array(
    [group_labels[np.where(group_ids == group_id)[0][0]] for group_id in train_val_groups],
    dtype=np.int32,
)
train_groups, val_groups = stratified_group_split(
    train_val_groups, train_val_group_labels, test_size=VAL_SIZE, random_state=RANDOM_STATE
)

train_samples = select_samples_by_groups(samples, train_groups)
val_samples = select_samples_by_groups(samples, val_groups)
test_samples = select_samples_by_groups(samples, test_groups)

print(
    "Split summary:"
    f" train_videos={len(train_groups)} val_videos={len(val_groups)} test_videos={len(test_groups)}"
)
print(
    "Sample summary:"
    f" train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}"
)

X_train, y_train = prepare_xy(train_samples)
X_val, y_val = prepare_xy(val_samples)
X_test, y_test = prepare_xy(test_samples)

model = build_model(feature_dim)

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
    validation_data=(X_val, y_val),
    callbacks=[checkpoint],
)

best_model_path = str(models_dir / "best_lstm_vit.h5")
best_model = load_model(best_model_path)

val_prob = best_model.predict(X_val, verbose=0).ravel()
threshold_info = choose_threshold(y_val, val_prob, target_specificity=TARGET_SPECIFICITY)
decision_threshold = float(threshold_info["threshold"])
print(
    "Selected decision threshold:"
    f" {decision_threshold:.4f} "
    f"(validation specificity={threshold_info['specificity']:.4f}, "
    f"balanced_accuracy={threshold_info['balanced_accuracy']:.4f})"
)

loss, acc = best_model.evaluate(X_test, y_test, verbose=0)
print(f"Best-model Test Loss: {loss:.4f}")
print(f"Best-model Test Accuracy: {acc:.4f}")

y_prob = best_model.predict(X_test, verbose=0).ravel()
threshold_metrics = metrics_at_threshold(y_test, y_prob, decision_threshold)
y_pred = threshold_metrics["y_pred"]

metrics = {
    "test_loss": float(loss),
    "test_accuracy": float(accuracy_score(y_test, y_pred)),
    "balanced_accuracy": float(threshold_metrics["balanced_accuracy"]),
    "precision": float(precision_score(y_test, y_pred, zero_division=0)),
    "recall_sensitivity": float(recall_score(y_test, y_pred, zero_division=0)),
    "specificity": float(threshold_metrics["specificity"]),
    "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, y_prob)),
    "pr_auc": float(average_precision_score(y_test, y_prob)),
    "mcc": float(matthews_corrcoef(y_test, y_pred)),
    "cohen_kappa": float(cohen_kappa_score(y_test, y_pred)),
    "brier_score": float(brier_score_loss(y_test, y_prob)),
    "decision_threshold": decision_threshold,
    "target_specificity": float(TARGET_SPECIFICITY),
    "split_strategy": "grouped_by_source_video",
    "split_counts": {
        "train_videos": int(len(train_groups)),
        "val_videos": int(len(val_groups)),
        "test_videos": int(len(test_groups)),
        "train_samples": int(len(train_samples)),
        "val_samples": int(len(val_samples)),
        "test_samples": int(len(test_samples)),
    },
    "validation_threshold_metrics": {
        key: value
        for key, value in threshold_info.items()
        if key != "y_pred"
    },
    "confusion_matrix": {
        "tn": int(threshold_metrics["tn"]),
        "fp": int(threshold_metrics["fp"]),
        "fn": int(threshold_metrics["fn"]),
        "tp": int(threshold_metrics["tp"]),
    },
}

# Bootstrap confidence intervals as a validity check on holdout performance.
rng = np.random.default_rng(RANDOM_STATE)
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
print(np.array([[threshold_metrics["tn"], threshold_metrics["fp"]], [threshold_metrics["fn"], threshold_metrics["tp"]]]))
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

with open(models_dir / "threshold_vit.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "decision_threshold": decision_threshold,
            "target_specificity": float(TARGET_SPECIFICITY),
            "validation_metrics": {
                key: value
                for key, value in threshold_info.items()
                if key != "y_pred"
            },
        },
        f,
        indent=2,
    )

print(f"\nSaved metrics to: {models_dir / 'metrics_vit.json'}")
print(f"Saved classification report to: {models_dir / 'classification_report_vit.txt'}")
print(f"Saved decision threshold to: {models_dir / 'threshold_vit.json'}")
