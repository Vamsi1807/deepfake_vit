from pathlib import Path

import cv2
import numpy as np

print("NumPy:", np.__version__)
print("OpenCV:", cv2.__version__)

print(cv2.getBuildInformation())

BASE_DIR = Path(__file__).resolve().parent
video_path = str(
    BASE_DIR.parent
    / "DeepFake Detection"
    / "Dataset"
    / "CelebDF"
    / "Celeb-real"
    / "id0_0001.mp4"
)

cap = cv2.VideoCapture(video_path)

print("Video opened:", cap.isOpened())
print("FPS:", cap.get(cv2.CAP_PROP_FPS))
print("Frame count:", cap.get(cv2.CAP_PROP_FRAME_COUNT))

ret, frame = cap.read()
print("First frame read:", ret)

cap.release()
