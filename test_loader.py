from pathlib import Path

from data_loader import load_celebdf, load_faceforensics

BASE_DIR = Path(__file__).resolve().parent
LEGACY_ROOT = BASE_DIR.parent / "DeepFake Detection" / "Dataset"

celeb_data = load_celebdf(str(LEGACY_ROOT / "CelebDF"))
ff_data = load_faceforensics(str(LEGACY_ROOT / "FaceForensics++_C23"))

print("CelebDF samples:", len(celeb_data))
print("FaceForensics++ samples:", len(ff_data))

print("Sample entry1:", celeb_data[0])
print("Sample entry2:", ff_data[0])
