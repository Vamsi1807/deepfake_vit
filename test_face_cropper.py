from pathlib import Path

from face_cropper import crop_faces_from_video

BASE_DIR = Path(__file__).resolve().parent
video_path = str(
    BASE_DIR.parent
    / "DeepFake Detection"
    / "Dataset"
    / "FaceForensics++_C23"
    / "DeepFakeDetection"
    / "01_11__meeting_serious__FCNL2OVP.mp4"
)

faces = crop_faces_from_video(video_path)
print(f"Cropped {len(faces)} faces")
