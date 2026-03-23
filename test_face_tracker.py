import os
from pathlib import Path

import cv2

from data_loader import load_celebdf
from face_detector import detect_faces
from face_tracker import FaceTracker
from frame_extractor import extract_frames

BASE_DIR = Path(__file__).resolve().parent
DATASET_ROOT = BASE_DIR.parent / "DeepFake Detection" / "Dataset" / "CelebDF"

video_data = load_celebdf(str(DATASET_ROOT))
NO_FACE_LOG = BASE_DIR / "no_face_videos_celebdf.txt"

MAX_PER_LABEL = 500


def has_saved_frames(video_dir: Path) -> bool:
    if not video_dir.exists():
        return False
    return any(video_dir.rglob("*.jpg"))


def count_existing_processed(label: str) -> int:
    label_root = BASE_DIR / "tracked_faces_celebdf" / label
    if not label_root.exists():
        return 0
    count = 0
    for video_dir in label_root.iterdir():
        if video_dir.is_dir() and has_saved_frames(video_dir):
            count += 1
    return count


def load_no_face_set(path: Path):
    if not path.exists():
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


real_count = count_existing_processed("real")
fake_count = count_existing_processed("fake")
no_face_seen = load_no_face_set(NO_FACE_LOG)
print(f"Resume mode: existing processed videos -> real={real_count}, fake={fake_count}")

for entry in video_data:
    label = "real" if entry["label"] == 0 else "fake"
    if (label == "real" and real_count >= MAX_PER_LABEL) or (
        label == "fake" and fake_count >= MAX_PER_LABEL
    ):
        continue

    video_path = entry["video_path"]
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    video_key = f"{label}/{video_id}"

    # Skip videos that were already processed in earlier runs.
    existing_video_dir = BASE_DIR / "tracked_faces_celebdf" / label / video_id
    if has_saved_frames(existing_video_dir):
        print(f"Skipping already processed: {video_key}")
        continue

    # Skip videos that already failed with no detected faces.
    if video_key in no_face_seen:
        print(f"Skipping known no-face video: {video_key}")
        continue

    print(f"Processing {video_key}...")

    frames = extract_frames(video_path)
    tracker = FaceTracker(max_distance=60, max_disappeared=30)
    video_has_face = False

    for i, frame in enumerate(frames):
        boxes = detect_faces(frame)
        tracked = tracker.update(boxes, frame=frame)

        for face_id, box in tracked.items():
            x, y, w, h = box
            face_crop = frame[y : y + h, x : x + w]
            if face_crop.size == 0:
                continue
            video_has_face = True
            save_dir = str(
                BASE_DIR
                / "tracked_faces_celebdf"
                / label
                / video_id
                / f"face_{face_id}"
            )
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"frame_{i}.jpg")
            cv2.imwrite(save_path, face_crop)

    if not video_has_face:
        print(f"No face detected in entire video: {video_key}. Skipping downstream steps.")
        if video_key not in no_face_seen:
            with open(NO_FACE_LOG, "a", encoding="utf-8") as f:
                f.write(f"{video_key}\n")
            no_face_seen.add(video_key)
        continue

    if label == "real":
        real_count += 1
    else:
        fake_count += 1

    if real_count >= MAX_PER_LABEL and fake_count >= MAX_PER_LABEL:
        break

print("Celeb-DF face tracking and saving complete.")
