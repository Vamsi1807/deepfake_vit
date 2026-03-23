import os
import json
from collections import deque
from pathlib import Path

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

APP_DIR = Path(__file__).resolve().parent
LSTM_MODEL_PATH = APP_DIR / "models" / "best_lstm_vit.h5"
lstm_model = load_model(str(LSTM_MODEL_PATH))

vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
vit_model = TFViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

MAX_SEQ_LENGTH = 30
DEFAULT_FAKE_THRESHOLD = 0.6
SMOOTHING_WINDOW = 5
MIN_FRAMES_FOR_PREDICTION = 5
BOX_SMOOTHING_ALPHA = 0.7
PRESENTATION_MODE = True
TRACK_HOLD_FRAMES = 3
SMOOTH_PLAYBACK = True
MIN_FAKE_CONSECUTIVE_FRAMES = 3
MIN_REAL_CONSECUTIVE_FRAMES = 2
MIN_FAKE_SEGMENT_SECONDS = 1.0
MIN_QUALIFIED_SEGMENTS_FOR_VIDEO_FAKE = 2
VIDEO_FAKE_SCORE_MARGIN = 0.08
MIN_SCORE_SAMPLES_FOR_FACE_VERDICT = 4
MIN_FACE_FRAME_RATIO_FOR_VERDICT = 0.20
MIN_TOTAL_FAKE_SECONDS_FOR_FACE_VERDICT = 2.0
MIN_DECISIVE_FACE_FRAME_RATIO = 0.35
MIN_DECISIVE_FACE_FRAMES = 15
MIN_DECISIVE_FAKE_FACES_FOR_VIDEO = 2

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


def load_fake_threshold():
    threshold_path = APP_DIR / "models" / "threshold_vit.json"
    if not threshold_path.exists():
        print(f"Threshold file not found. Using default threshold {DEFAULT_FAKE_THRESHOLD:.2f}")
        return DEFAULT_FAKE_THRESHOLD
    try:
        with open(threshold_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        threshold = float(payload.get("decision_threshold", DEFAULT_FAKE_THRESHOLD))
        print(f"Loaded calibrated fake threshold: {threshold:.4f}")
        return threshold
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"Failed to load threshold file ({exc}). Using default threshold {DEFAULT_FAKE_THRESHOLD:.2f}")
        return DEFAULT_FAKE_THRESHOLD


FAKE_THRESHOLD = load_fake_threshold()


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
            tracker.update(boxes, frame=frame)
        else:
            # Advance tracker state even when detection is skipped.
            tracker.update([], frame=frame)

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
                    "frame_count": 0,
                    "is_fake": False,
                    "fake_streak": 0,
                    "real_streak": 0,
                    "active_segment": None,
                }

            predictions[face_id]["frame_count"] += 1
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
                over_threshold = avg_score >= FAKE_THRESHOLD

                if over_threshold:
                    predictions[face_id]["fake_streak"] += 1
                    predictions[face_id]["real_streak"] = 0
                else:
                    predictions[face_id]["real_streak"] += 1
                    predictions[face_id]["fake_streak"] = 0

                if (
                    over_threshold
                    and not predictions[face_id]["is_fake"]
                    and predictions[face_id]["fake_streak"] >= MIN_FAKE_CONSECUTIVE_FRAMES
                ):
                    predictions[face_id]["active_segment"] = {
                        "face_id": int(face_id),
                        "start_frame": int(i),
                        "start_time_sec": float(i * seconds_per_frame),
                        "peak_fake_score": float(avg_score),
                        "reason_signals": build_reason_signals(
                            avg_score, len(predictions[face_id]["features"])
                        ),
                    }
                    predictions[face_id]["is_fake"] = True
                elif predictions[face_id]["is_fake"] and predictions[face_id]["active_segment"] is not None:
                    predictions[face_id]["active_segment"]["peak_fake_score"] = float(
                        max(
                            predictions[face_id]["active_segment"]["peak_fake_score"],
                            avg_score,
                        )
                    )

                if (
                    (not over_threshold)
                    and predictions[face_id]["is_fake"]
                    and predictions[face_id]["real_streak"] >= MIN_REAL_CONSECUTIVE_FRAMES
                ):
                    active = predictions[face_id]["active_segment"]
                    if active is not None:
                        active["end_frame"] = int(i)
                        active["end_time_sec"] = float(i * seconds_per_frame)
                        duration = active["end_time_sec"] - active["start_time_sec"]
                        if duration >= MIN_FAKE_SEGMENT_SECONDS:
                            analysis_segments.append(active)
                        predictions[face_id]["active_segment"] = None
                    predictions[face_id]["is_fake"] = False

                is_fake_now = predictions[face_id]["is_fake"]
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
            duration = active["end_time_sec"] - active["start_time_sec"]
            if duration >= MIN_FAKE_SEGMENT_SECONDS:
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

    fake_segments_by_face = {}
    for segment in analysis_segments:
        fake_segments_by_face.setdefault(int(segment["face_id"]), []).append(segment)

    face_verdicts = {}
    fake_face_ids = []
    for face_id, state in predictions.items():
        scores = list(state["scores"])
        avg_score = float(sum(scores) / len(scores)) if scores else 0.0
        score_samples = int(len(scores))
        frame_count = int(state.get("frame_count", 0))
        frame_ratio = frame_count / max(float(len(frames)), 1.0)
        segments = fake_segments_by_face.get(int(face_id), [])
        qualified_segments = int(len(segments))
        total_fake_seconds = float(
            sum(
                max(
                    0.0,
                    float(segment.get("end_time_sec", 0.0))
                    - float(segment.get("start_time_sec", 0.0)),
                )
                for segment in segments
            )
        )
        has_valid_fake_segment = qualified_segments > 0
        has_reliable_signal = (
            score_samples >= MIN_SCORE_SAMPLES_FOR_FACE_VERDICT
            and frame_ratio >= MIN_FACE_FRAME_RATIO_FOR_VERDICT
            and total_fake_seconds >= MIN_TOTAL_FAKE_SECONDS_FOR_FACE_VERDICT
        )
        passes_segment_rule = qualified_segments >= MIN_QUALIFIED_SEGMENTS_FOR_VIDEO_FAKE
        passes_score_rule = avg_score >= (FAKE_THRESHOLD + VIDEO_FAKE_SCORE_MARGIN)
        is_fake = bool(
            has_valid_fake_segment
            and has_reliable_signal
            and (passes_segment_rule or passes_score_rule)
        )
        face_verdicts[int(face_id)] = {
            "verdict": "FAKE" if is_fake else "REAL",
            "avg_fake_score": avg_score,
            "score_samples": score_samples,
            "frames_seen": frame_count,
            "frame_ratio": float(frame_ratio),
            "qualified_fake_segments": qualified_segments,
            "total_fake_seconds": total_fake_seconds,
        }
        if is_fake:
            fake_face_ids.append(int(face_id))
    fake_face_ids.sort()

    decisive_fake_face_ids = []
    for face_id in fake_face_ids:
        verdict = face_verdicts.get(face_id, {})
        frames_seen = int(verdict.get("frames_seen", 0))
        frame_ratio = float(verdict.get("frame_ratio", 0.0))
        if (
            frame_ratio >= MIN_DECISIVE_FACE_FRAME_RATIO
            or frames_seen >= MIN_DECISIVE_FACE_FRAMES
        ):
            decisive_fake_face_ids.append(int(face_id))
    decisive_fake_face_ids.sort()

    primary_face_id = None
    if face_verdicts:
        primary_face_id = int(
            max(
                face_verdicts.keys(),
                key=lambda k: int(face_verdicts[k].get("frames_seen", 0)),
            )
        )

    primary_face_is_fake = (
        primary_face_id is not None and primary_face_id in decisive_fake_face_ids
    )
    detected_face_ids = sorted(int(face_id) for face_id in seen_face_ids)
    video_verdict = (
        "FAKE"
        if (
            primary_face_is_fake
            or len(decisive_fake_face_ids) >= MIN_DECISIVE_FAKE_FACES_FOR_VIDEO
        )
        else "REAL"
    )

    analysis_path = os.path.splitext(video_path)[0] + "_analysis.json"
    analysis_payload = {
        "video_path": video_path,
        "annotated_video_path": output_path,
        "status": "ok",
        "total_frames_processed": int(len(frames)),
        "inference_fps": float(output_fps),
        "detected_face_ids": detected_face_ids,
        "video_verdict": video_verdict,
        "fake_face_ids": decisive_fake_face_ids,
        "candidate_fake_face_ids": fake_face_ids,
        "primary_face_id": primary_face_id,
        "face_verdicts": face_verdicts,
        "fake_threshold": float(FAKE_THRESHOLD),
        "min_fake_consecutive_frames": int(MIN_FAKE_CONSECUTIVE_FRAMES),
        "min_fake_segment_seconds": float(MIN_FAKE_SEGMENT_SECONDS),
        "min_score_samples_for_face_verdict": int(MIN_SCORE_SAMPLES_FOR_FACE_VERDICT),
        "min_face_frame_ratio_for_verdict": float(MIN_FACE_FRAME_RATIO_FOR_VERDICT),
        "min_total_fake_seconds_for_face_verdict": float(MIN_TOTAL_FAKE_SECONDS_FOR_FACE_VERDICT),
        "min_decisive_face_frame_ratio": float(MIN_DECISIVE_FACE_FRAME_RATIO),
        "min_decisive_face_frames": int(MIN_DECISIVE_FACE_FRAMES),
        "min_decisive_fake_faces_for_video": int(MIN_DECISIVE_FAKE_FACES_FOR_VIDEO),
        "note": "Reason signals are model-based indicators, not forensic proof.",
        "fake_segments": analysis_segments,
    }
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis_payload, f, indent=2)

    print(f"Annotated output saved to: {output_path}")
    print(f"Video verdict: {video_verdict}")
    print(f"Detected face IDs: {detected_face_ids}")
    if decisive_fake_face_ids:
        print(f"Fake face IDs: {decisive_fake_face_ids}")
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
