from mtcnn import MTCNN

detector = MTCNN()

def detect_faces(frame):
    """
    Input: frame (numpy array)
    Output: list of bounding boxes [(x, y, w, h), ...]
    """
    detections = detector.detect_faces(frame)
    boxes = []

    for d in detections:
        x, y, w, h = d['box']
        x, y = max(0, x), max(0, y)
        boxes.append((x, y, w, h))

    return boxes
