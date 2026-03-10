import os

def load_faceforensics(root):
    data = []
    manipulation_folders = [
        "DeepFakeDetection", "FaceSwap", "FaceShifter",
        "Face2Face", "NeuralTextures", "Deepfakes"
    ]

    # Real videos (label 0)
    real_path = os.path.join(root, "original")
    if os.path.exists(real_path):
        for f in os.listdir(real_path):
            if f.endswith(".mp4"):
                data.append({
                    "video_path": os.path.join(real_path, f),
                    "label": 0,
                    "dataset": "FF++"
                })

    # Fake videos (label 1)
    for folder in manipulation_folders:
        folder_path = os.path.join(root, folder)
        if not os.path.exists(folder_path):
            continue
        for f in os.listdir(folder_path):
            if f.endswith(".mp4"):
                data.append({
                    "video_path": os.path.join(folder_path, f),
                    "label": 1,
                    "dataset": "FF++"
                })

    return data

def load_celebdf(root):
    data = []
    real_path = os.path.join(root, 'Celeb-real')
    fake_path = os.path.join(root, 'Celeb-synthesis')

    if os.path.exists(real_path):
        for f in os.listdir(real_path):
            if f.endswith('.mp4'):
                data.append({
                    "video_path": os.path.join(real_path, f),
                    "label": 0,
                    "dataset": "CelebDF"
                })

    if os.path.exists(fake_path):
        for f in os.listdir(fake_path):
            if f.endswith('.mp4'):
                data.append({
                    "video_path": os.path.join(fake_path, f),
                    "label": 1,
                    "dataset": "CelebDF"
                })

    return data