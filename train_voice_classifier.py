import os
import pickle
import json
import random
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from feature_extractor import extract_dataset_features, FEATURE_NAMES


DATASET_DIR = Path("voice_dataset")
MODEL_DIR = Path("dataset_model")
MODEL_DIR.mkdir(exist_ok=True)

AUDIO_EXTS = {".wav", ".mp3", ".ogg", ".webm", ".m4a", ".flac"}
MAX_FILES_PER_CLASS = int(os.environ.get("VOICE_MAX_FILES_PER_CLASS", "150"))


def collect_audio_files():
    folders = {
        0: [DATASET_DIR / "healthy", DATASET_DIR / "Healthy", DATASET_DIR / "normal", DATASET_DIR / "Normal"],
        1: [DATASET_DIR / "parkinson", DATASET_DIR / "Parkinson", DATASET_DIR / "parkinsons", DATASET_DIR / "Parkinsons"],
    }

    samples = []
    random.seed(42)
    for label, candidates in folders.items():
        folder = next((p for p in candidates if p.exists()), None)
        if folder is None:
            raise FileNotFoundError(
                f"Missing folder for label {label}: tried {', '.join(str(p) for p in candidates)}\n"
                "Expected structure:\n"
                "voice_dataset/healthy/*.wav\n"
                "voice_dataset/parkinson/*.wav\n"
                "Your folder names can also be Healthy and Parkinsons."
            )
        class_files = [
            path for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in AUDIO_EXTS
        ]
        class_files = sorted(class_files)
        if MAX_FILES_PER_CLASS > 0 and len(class_files) > MAX_FILES_PER_CLASS:
            class_files = random.sample(class_files, MAX_FILES_PER_CLASS)
            class_files = sorted(class_files)
        print(f"Using {len(class_files)} files from {folder}")
        for path in class_files:
            if path.is_file() and path.suffix.lower() in AUDIO_EXTS:
                samples.append((path, label))

    if not samples:
        raise RuntimeError("No audio files found in voice_dataset/healthy or voice_dataset/parkinson.")
    return samples


def build_feature_matrix(samples):
    X, y, used_files, skipped_files = [], [], [], []

    for path, label in samples:
        print(f"Extracting: {path}")
        features = extract_dataset_features(str(path))
        if not features:
            skipped_files.append(str(path))
            print(f"  skipped: could not extract features")
            continue

        X.append([features[name] for name in FEATURE_NAMES])
        y.append(label)
        used_files.append(str(path))

    if len(X) < 4:
        raise RuntimeError("Not enough valid audio files. Add more clear healthy and Parkinson WAV samples.")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), used_files, skipped_files


def risk_from_probability(p_pd):
    if p_pd >= 0.75:
        return "High"
    if p_pd >= 0.55:
        return "Moderate"
    if p_pd >= 0.40:
        return "Low"
    return "Normal"


def main():
    samples = collect_audio_files()
    X, y, used_files, skipped_files = build_feature_matrix(samples)

    classes, counts = np.unique(y, return_counts=True)
    print("Class counts:", dict(zip(classes.tolist(), counts.tolist())))
    if len(classes) < 2:
        raise RuntimeError("Need both classes: healthy and parkinson.")

    stratify = y if min(counts) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    k = min(5, len(X_train))
    clf = VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")),
            ("gb", GradientBoostingClassifier(random_state=42)),
            ("knn", KNeighborsClassifier(n_neighbors=max(1, k))),
        ],
        voting="soft",
    )
    clf.fit(X_train_sc, y_train)

    pred = clf.predict(X_test_sc)
    proba = clf.predict_proba(X_test_sc)
    acc = accuracy_score(y_test, pred)

    print("\nAccuracy:", round(acc, 4))
    print("\nClassification report:")
    print(classification_report(y_test, pred, target_names=["Healthy", "Parkinson"], zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, pred).tolist())

    with open(MODEL_DIR / "voice_classifier.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open(MODEL_DIR / "voice_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(MODEL_DIR / "voice_features.pkl", "wb") as f:
        pickle.dump(FEATURE_NAMES, f)

    metrics = {
        "accuracy": float(acc),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "total_valid_samples": int(len(X)),
        "skipped_files": skipped_files,
        "features": FEATURE_NAMES,
        "risk_mapping": {
            "0.00-0.39": "Normal",
            "0.40-0.54": "Low",
            "0.55-0.74": "Moderate",
            "0.75-1.00": "High",
        },
    }
    with open(MODEL_DIR / "voice_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved:")
    print(MODEL_DIR / "voice_classifier.pkl")
    print(MODEL_DIR / "voice_scaler.pkl")
    print(MODEL_DIR / "voice_features.pkl")
    print(MODEL_DIR / "voice_metrics.json")


if __name__ == "__main__":
    main()
