from pathlib import Path

import cv2

from face_detector import detect_faces
from frame_extractor import extract_frames

BASE_DIR = Path(__file__).resolve().parent
video_path = str(
    BASE_DIR.parent
    / "DeepFake Detection"
    / "Dataset"
    / "FaceForensics++_C23"
    / "DeepFakeDetection"
    / "01_11__meeting_serious__FCNL2OVP.mp4"
)

frames = extract_frames(video_path)
print(f"Extracted {len(frames)} frames")

for i, frame in enumerate(frames[:5]):
    boxes = detect_faces(frame)
    print(f"Frame {i}: Detected {len(boxes)} faces")

    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow(f"Frame {i}", frame)
    cv2.waitKey(500)

cv2.destroyAllWindows()
