import os
import json
from collections import deque

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from face_detector import detect_faces
from face_tracker import FaceTracker
from frame_extractor import extract_frames

try:
    from transformers import TFViTModel, ViTImageProcessor
except ImportError as exc:
    raise ImportError(
        "transformers is required for ViT inference. Install with: pip install transformers"
    ) from exc

def configure_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as exc:
                # Happens if TF already initialized; safe to continue.
                print(f"GPU memory growth already fixed/initialized: {exc}")
        print(f"GPU detected and enabled: {[gpu.name for gpu in gpus]}")
    else:
        print("No TensorFlow GPU detected. Inference will run on CPU.")


# Configure device policy before loading any TF/Keras models.
configure_gpu()

LSTM_MODEL_PATH = "models/best_lstm_vit.h5"
lstm_model = load_model(LSTM_MODEL_PATH)

vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
vit_model = TFViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

MAX_SEQ_LENGTH = 30
FAKE_THRESHOLD = 0.6
SMOOTHING_WINDOW = 5
MIN_FRAMES_FOR_PREDICTION = 5
BOX_SMOOTHING_ALPHA = 0.7
PRESENTATION_MODE = True
TRACK_HOLD_FRAMES = 3
SMOOTH_PLAYBACK = True

if PRESENTATION_MODE:
    # Fast profile for demos/live usage.
    INFER_FPS = 6
    DETECT_EVERY_N_FRAMES = 3
    LSTM_PREDICT_EVERY_N_FRAMES = 2
    INFER_MAX_SECONDS = None
else:
    # Full profile for best quality.
    INFER_FPS = 12
    DETECT_EVERY_N_FRAMES = 1
    LSTM_PREDICT_EVERY_N_FRAMES = 1
    INFER_MAX_SECONDS = None


def build_reason_signals(score, feature_count):
    signals = []
    if score >= 0.85:
        signals.append("High model confidence for manipulated facial dynamics.")
    elif score >= FAKE_THRESHOLD:
        signals.append("Consistent manipulation pattern detected across recent frames.")

    if feature_count < MAX_SEQ_LENGTH:
        signals.append("Decision based on partial temporal window; confidence may stabilize with more frames.")
    else:
        signals.append("Decision supported by full temporal context window.")

    return signals


def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 25.0


def get_video_duration_seconds(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps and fps > 0 and frame_count and frame_count > 0:
        return float(frame_count / fps)
    return 0.0


def filter_overlapping_boxes(boxes, iou_threshold=0.5):
    if not boxes:
        return []

    boxes_arr = np.array(boxes)
    areas = boxes_arr[:, 2] * boxes_arr[:, 3]
    order = np.argsort(-areas)
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(boxes[i])

        if len(order) == 1:
            break

        x1_i, y1_i, w1_i, h1_i = boxes_arr[i]
        x2_i, y2_i = x1_i + w1_i, y1_i + h1_i

        rest_boxes = boxes_arr[order[1:]]
        x1_rest = rest_boxes[:, 0]
        y1_rest = rest_boxes[:, 1]
        x2_rest = x1_rest + rest_boxes[:, 2]
        y2_rest = y1_rest + rest_boxes[:, 3]

        xx1 = np.maximum(x1_i, x1_rest)
        yy1 = np.maximum(y1_i, y1_rest)
        xx2 = np.minimum(x2_i, x2_rest)
        yy2 = np.minimum(y2_i, y2_rest)

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        intersection = w * h

        area_i = w1_i * h1_i
        area_rest = rest_boxes[:, 2] * rest_boxes[:, 3]
        union = area_i + area_rest - intersection

        iou = intersection / union
        order = order[1:][iou < iou_threshold]

    return keep


def extract_vit_features_batch(face_crops_bgr):
    if not face_crops_bgr:
        return np.empty((0, 768), dtype=np.float32)

    face_batch_rgb = []
    for crop in face_crops_bgr:
        face_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        face_batch_rgb.append(cv2.resize(face_rgb, (224, 224)))

    inputs = vit_processor(images=face_batch_rgb, return_tensors="tf")
    outputs = vit_model(**inputs, training=False)
    return outputs.last_hidden_state[:, 0, :].numpy()


def process_and_annotate_video(video_path):
    frames = extract_frames(video_path, target_fps=INFER_FPS, max_seconds=INFER_MAX_SECONDS)
    input_duration = get_video_duration_seconds(video_path)
    original_fps = get_video_fps(video_path)

    tracker = FaceTracker()
    output_path = os.path.splitext(video_path)[0] + "_output.mp4"

    if len(frames) == 0:
        print("Error: No frames extracted")
        return None

    height, width, _ = frames[0].shape
    if SMOOTH_PLAYBACK and original_fps > 0:
        output_fps = max(float(original_fps), 1.0)
    elif input_duration > 0 and len(frames) > 0:
        output_fps = max(len(frames) / input_duration, 1.0)
    else:
        output_fps = float(INFER_FPS)

    if input_duration > 0 and len(frames) > 0:
        seconds_per_frame = float(input_duration / len(frames))
    else:
        seconds_per_frame = 1.0 / max(float(INFER_FPS), 1.0)

    if input_duration > 0 and output_fps > 0:
        target_output_frames = max(1, int(round(input_duration * output_fps)))
    else:
        target_output_frames = len(frames)

    # Prefer browser-friendly MP4 encoders first; fallback to mp4v.
    writer = None
    for codec in ("avc1", "H264", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        candidate = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
        if candidate.isOpened():
            writer = candidate
            print(f"Using video codec: {codec}")
            break
        candidate.release()

    if writer is None:
        print("Error: Could not open video writer for annotated output.")
        return None
    predictions = {}
    smoothed_boxes = {}
    analysis_segments = []
    seen_face_ids = set()
    any_face_detected = False

    for i, frame in enumerate(frames):
        if i % DETECT_EVERY_N_FRAMES == 0:
            boxes = detect_faces(frame)
            boxes = filter_overlapping_boxes(boxes, iou_threshold=0.5)
            tracker.update(boxes)
        else:
            # Advance tracker state even when detection is skipped.
            tracker.update([])

        # Keep recent tracks visible between detection frames to avoid label flicker.
        tracked = {
            face_id: box
            for face_id, (box, _) in tracker.tracks.items()
            if tracker.disappeared.get(face_id, 0) <= TRACK_HOLD_FRAMES
        }

        face_ids = []
        face_crops = []
        face_draw_boxes = {}

        for face_id, box in tracked.items():
            x, y, w, h = box
            if w * h < 4000:
                continue

            if face_id in smoothed_boxes:
                px, py, pw, ph = smoothed_boxes[face_id]
                x = int(BOX_SMOOTHING_ALPHA * x + (1.0 - BOX_SMOOTHING_ALPHA) * px)
                y = int(BOX_SMOOTHING_ALPHA * y + (1.0 - BOX_SMOOTHING_ALPHA) * py)
                w = int(BOX_SMOOTHING_ALPHA * w + (1.0 - BOX_SMOOTHING_ALPHA) * pw)
                h = int(BOX_SMOOTHING_ALPHA * h + (1.0 - BOX_SMOOTHING_ALPHA) * ph)
            smoothed_boxes[face_id] = (x, y, w, h)

            face_crop = frame[y : y + h, x : x + w]
            if face_crop.size == 0:
                continue

            face_ids.append(face_id)
            face_crops.append(face_crop)
            face_draw_boxes[face_id] = (x, y, w, h)

        batch_feats = extract_vit_features_batch(face_crops)

        for idx, face_id in enumerate(face_ids):
            feat = batch_feats[idx]
            any_face_detected = True
            seen_face_ids.add(int(face_id))

            if face_id not in predictions:
                predictions[face_id] = {
                    "features": deque(maxlen=MAX_SEQ_LENGTH),
                    "scores": deque(maxlen=SMOOTHING_WINDOW),
                    "is_fake": False,
                    "active_segment": None,
                }

            predictions[face_id]["features"].append(feat)

            if (
                len(predictions[face_id]["features"]) >= MIN_FRAMES_FOR_PREDICTION
                and i % LSTM_PREDICT_EVERY_N_FRAMES == 0
            ):
                seq = pad_sequences(
                    [list(predictions[face_id]["features"])],
                    maxlen=MAX_SEQ_LENGTH,
                    dtype="float32",
                    padding="post",
                    truncating="post",
                )
                pred_score = lstm_model.predict(seq, verbose=0)[0][0]
                predictions[face_id]["scores"].append(pred_score)

            if predictions[face_id]["scores"]:
                avg_score = sum(predictions[face_id]["scores"]) / len(
                    predictions[face_id]["scores"]
                )
                is_fake_now = avg_score > FAKE_THRESHOLD

                if is_fake_now and not predictions[face_id]["is_fake"]:
                    predictions[face_id]["active_segment"] = {
                        "face_id": int(face_id),
                        "start_frame": int(i),
                        "start_time_sec": float(i * seconds_per_frame),
                        "peak_fake_score": float(avg_score),
                        "reason_signals": build_reason_signals(
                            avg_score, len(predictions[face_id]["features"])
                        ),
                    }
                elif is_fake_now and predictions[face_id]["active_segment"] is not None:
                    predictions[face_id]["active_segment"]["peak_fake_score"] = float(
                        max(
                            predictions[face_id]["active_segment"]["peak_fake_score"],
                            avg_score,
                        )
                    )

                if (not is_fake_now) and predictions[face_id]["is_fake"]:
                    active = predictions[face_id]["active_segment"]
                    if active is not None:
                        active["end_frame"] = int(i)
                        active["end_time_sec"] = float(i * seconds_per_frame)
                        analysis_segments.append(active)
                        predictions[face_id]["active_segment"] = None

                predictions[face_id]["is_fake"] = is_fake_now
                label = "FAKE" if is_fake_now else "REAL"
                color = (0, 0, 255) if is_fake_now else (0, 255, 0)
            else:
                label = "ANALYZING"
                color = (255, 255, 0)

            x, y, w, h = face_draw_boxes[face_id]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        start_out_idx = int(round((i * target_output_frames) / len(frames)))
        end_out_idx = int(round(((i + 1) * target_output_frames) / len(frames)))
        repeats = max(1, end_out_idx - start_out_idx)
        for _ in range(repeats):
            writer.write(frame)

    writer.release()

    for face_id, state in predictions.items():
        if state["is_fake"] and state["active_segment"] is not None:
            active = state["active_segment"]
            active["end_frame"] = int(len(frames) - 1)
            active["end_time_sec"] = float((len(frames) - 1) * seconds_per_frame)
            analysis_segments.append(active)

    if not any_face_detected:
        analysis_path = os.path.splitext(video_path)[0] + "_analysis.json"
        analysis_payload = {
            "video_path": video_path,
            "annotated_video_path": None,
            "status": "no_face_detected",
            "message": "No face detected in the video. Processing stopped.",
            "total_frames_processed": int(len(frames)),
            "inference_fps": float(output_fps),
            "detected_face_ids": [],
            "video_verdict": "NO_FACE_DETECTED",
            "fake_face_ids": [],
            "face_verdicts": {},
            "fake_threshold": float(FAKE_THRESHOLD),
            "note": "Reason signals are model-based indicators, not forensic proof.",
            "fake_segments": [],
        }
        with open(analysis_path, "w", encoding="utf-8") as f:
            json.dump(analysis_payload, f, indent=2)

        if os.path.exists(output_path):
            os.remove(output_path)

        print("No face detected in entire video. Processing stopped.")
        print(f"Analysis JSON saved to: {analysis_path}")
        return {
            "status": "no_face_detected",
            "annotated_video_path": None,
            "analysis_json_path": analysis_path,
        }

    face_verdicts = {}
    fake_face_ids = []
    for face_id, state in predictions.items():
        scores = list(state["scores"])
        avg_score = float(sum(scores) / len(scores)) if scores else 0.0
        is_fake = bool(scores and avg_score > FAKE_THRESHOLD)
        face_verdicts[int(face_id)] = {
            "verdict": "FAKE" if is_fake else "REAL",
            "avg_fake_score": avg_score,
            "score_samples": int(len(scores)),
        }
        if is_fake:
            fake_face_ids.append(int(face_id))
    fake_face_ids.sort()
    detected_face_ids = sorted(int(face_id) for face_id in seen_face_ids)
    video_verdict = "FAKE" if fake_face_ids else "REAL"

    analysis_path = os.path.splitext(video_path)[0] + "_analysis.json"
    analysis_payload = {
        "video_path": video_path,
        "annotated_video_path": output_path,
        "status": "ok",
        "total_frames_processed": int(len(frames)),
        "inference_fps": float(output_fps),
        "detected_face_ids": detected_face_ids,
        "video_verdict": video_verdict,
        "fake_face_ids": fake_face_ids,
        "face_verdicts": face_verdicts,
        "fake_threshold": float(FAKE_THRESHOLD),
        "note": "Reason signals are model-based indicators, not forensic proof.",
        "fake_segments": analysis_segments,
    }
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis_payload, f, indent=2)

    print(f"Annotated output saved to: {output_path}")
    print(f"Video verdict: {video_verdict}")
    print(f"Detected face IDs: {detected_face_ids}")
    if fake_face_ids:
        print(f"Fake face IDs: {fake_face_ids}")
    else:
        print("Fake face IDs: []")
    print(f"Analysis JSON saved to: {analysis_path}")
    return {
        "status": "ok",
        "annotated_video_path": output_path,
        "analysis_json_path": analysis_path,
    }


if __name__ == "__main__":
    video_path = r"C:\Users\vamsi\Pictures\Camera Roll\WIN_20260308_20_10_20_Pro.mp4"
    process_and_annotate_video(video_path)
