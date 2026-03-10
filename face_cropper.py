import cv2

def crop_face_with_margin(frame, bbox, margin=0.2):
    x, y, w, h = bbox

    h_img, w_img, _ = frame.shape

    x_margin = int(w * margin)
    y_margin = int(h * margin)

    x1 = max(0, x - x_margin)
    y1 = max(0, y - y_margin)
    x2 = min(w_img, x + w + x_margin)
    y2 = min(h_img, y + h + y_margin)

    face = frame[y1:y2, x1:x2]
    face = cv2.resize(face, (224, 224))

    return face
