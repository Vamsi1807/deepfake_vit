import os
from pathlib import Path

import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import UnidentifiedImageError

try:
    from transformers import TFViTModel, ViTImageProcessor
except ImportError as exc:
    raise ImportError(
        "transformers is required for ViT feature extraction. Install with: pip install transformers"
    ) from exc


BASE_DIR = Path(__file__).resolve().parent

# ViT-B/16 produces 768-dim CLS embeddings.
IMAGE_SIZE = 224
vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
vit_model = TFViTModel.from_pretrained("google/vit-base-patch16-224-in21k")


def extract_sequence_features(face_dir, batch_size=16):
    image_paths = [
        os.path.join(face_dir, f)
        for f in sorted(os.listdir(face_dir))
        if f.lower().endswith(".jpg")
    ]
    embeddings = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch_images = []

        for path in batch_paths:
            try:
                img = load_img(path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
                x = img_to_array(img).astype(np.uint8)
                batch_images.append(x)
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                print(f"Skipping unreadable image: {path} ({exc})")
                continue

        if not batch_images:
            continue

        inputs = vit_processor(images=batch_images, return_tensors="tf")
        outputs = vit_model(**inputs, training=False)
        batch_features = outputs.last_hidden_state[:, 0, :].numpy()
        embeddings.extend(batch_features)

    return np.array(embeddings, dtype=np.float32)


def save_feature_sequence(face_dir, out_path):
    features = extract_sequence_features(face_dir)
    if features.size == 0:
        print(f"No valid frames in {face_dir}. Skipping feature file.")
        return False
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, features)
    print(f"Saved {features.shape} features to {out_path}")
    return True


def process_dataset(input_root, output_root, dataset_name, max_videos_per_label=25):
    for label in ["real", "fake"]:
        label_path = os.path.join(input_root, label)
        if not os.path.exists(label_path):
            continue
        processed = 0
        video_names = sorted(
            [v for v in os.listdir(label_path) if os.path.isdir(os.path.join(label_path, v))]
        )

        for video_name in video_names:
            if processed >= max_videos_per_label:
                break
            video_path = os.path.join(label_path, video_name)
            video_created_features = False

            for face_id in os.listdir(video_path):
                face_path = os.path.join(video_path, face_id)
                if not os.path.isdir(face_path):
                    continue
                out_path = os.path.join(output_root, dataset_name, label, f"{video_name}_{face_id}.npy")
                if os.path.exists(out_path):
                    continue
                was_saved = save_feature_sequence(face_path, out_path)
                if was_saved:
                    video_created_features = True

            if video_created_features:
                processed += 1
            else:
                print(f"Skipped {video_name} (already processed)")


def process_multiple_datasets(datasets, output_root="extracted_features_vit", max_videos_per_label=25):
    for dataset_name, input_root in datasets.items():
        if not os.path.exists(input_root):
            print(f"Skipping {dataset_name}: {input_root} not found")
            continue
        print(f"Processing {dataset_name} from {input_root}")
        process_dataset(
            input_root=input_root,
            output_root=output_root,
            dataset_name=dataset_name,
            max_videos_per_label=max_videos_per_label,
        )


if __name__ == "__main__":
    datasets = {
        "ffpp": str(BASE_DIR / "tracked_faces"),
        "celebdf": str(BASE_DIR / "tracked_faces_celebdf"),
    }
    output_root = str(BASE_DIR / "extracted_features_vit")
    process_multiple_datasets(datasets, output_root=output_root, max_videos_per_label=100)
