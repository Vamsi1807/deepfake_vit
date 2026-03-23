import warnings

try:
    from retinaface import RetinaFace
except ImportError:
    RetinaFace = None

try:
    from mtcnn import MTCNN
except ImportError:
    MTCNN = None


_backend = None
_mtcnn_detector = None


def _load_backend():
    global _backend, _mtcnn_detector

    if RetinaFace is not None:
        _backend = "retinaface"
        print("Face detector backend: RetinaFace")
        return

    if MTCNN is not None:
        _mtcnn_detector = MTCNN()
        _backend = "mtcnn"
        print("Face detector backend: MTCNN (fallback)")
        warnings.warn(
            "RetinaFace is not installed; falling back to MTCNN. "
            "Tracking quality may be lower in multi-face videos.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    raise ImportError(
        "No supported face detector is installed. Install `retinaface` for the preferred "
        "GPU-first detector or `mtcnn` as a fallback."
    )


def _clip_box(x, y, w, h, frame_shape):
    height, width = frame_shape[:2]
    x = max(0, int(round(x)))
    y = max(0, int(round(y)))
    w = int(round(w))
    h = int(round(h))
    if w <= 0 or h <= 0:
        return None
    w = min(w, width - x)
    h = min(h, height - y)
    if w <= 0 or h <= 0:
        return None
    return (x, y, w, h)


def _detect_with_retinaface(frame):
    detections = RetinaFace.detect_faces(frame)
    if not detections:
        return []
    if isinstance(detections, dict) and "face_1" not in detections and "score" in detections:
        detections = {"face_1": detections}

    boxes = []
    for detection in detections.values():
        facial_area = detection.get("facial_area")
        if facial_area is None or len(facial_area) != 4:
            continue
        x1, y1, x2, y2 = facial_area
        box = _clip_box(x1, y1, x2 - x1, y2 - y1, frame.shape)
        if box is not None:
            boxes.append(box)
    return boxes


def _detect_with_mtcnn(frame):
    detections = _mtcnn_detector.detect_faces(frame)
    boxes = []
    for detection in detections:
        box = detection.get("box")
        if box is None or len(box) != 4:
            continue
        clipped = _clip_box(*box, frame.shape)
        if clipped is not None:
            boxes.append(clipped)
    return boxes


def detect_faces(frame):
    """
    Input: frame (numpy array)
    Output: list of bounding boxes [(x, y, w, h), ...]
    """
    if _backend is None:
        _load_backend()

    if _backend == "retinaface":
        return _detect_with_retinaface(frame)
    return _detect_with_mtcnn(frame)
