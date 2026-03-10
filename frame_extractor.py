import cv2

def extract_frames(video_path, target_fps=10, max_seconds=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Cannot open video:", video_path)
        return []

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps is None or original_fps <= 0:
        original_fps = target_fps  # fallback

    frame_interval = max(int(original_fps / target_fps), 1)
    max_frames = None if max_seconds is None else target_fps * max_seconds

    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frames.append(frame)

        if max_frames is not None and len(frames) >= max_frames:
            break

        frame_count += 1

    cap.release()
    return frames
