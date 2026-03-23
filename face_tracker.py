from collections import defaultdict

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import euclidean


class FaceTracker:
    def __init__(
        self,
        max_distance=60,
        max_disappeared=30,
        reid_ttl=45,
        appearance_weight=0.45,
        min_iou_for_match=0.05,
    ):
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared
        self.reid_ttl = reid_ttl
        self.appearance_weight = appearance_weight
        self.min_iou_for_match = min_iou_for_match
        self.next_id = 0
        self.tracks = {}  # id -> (box, centroid)
        self.disappeared = defaultdict(int)
        self.track_features = {}
        self.archived_tracks = {}

    def _centroid(self, box):
        x, y, w, h = box
        return (x + w // 2, y + h // 2)

    def _iou(self, box_a, box_b):
        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b

        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh

        inter_x1 = max(ax, bx)
        inter_y1 = max(ay, by)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = aw * ah
        area_b = bw * bh
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union

    def _extract_feature(self, frame, box):
        x, y, w, h = box
        face = frame[y : y + h, x : x + w]
        if face.size == 0:
            return None

        face = cv2.resize(face, (64, 64))
        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [12, 12], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (16, 16)).astype(np.float32).flatten()
        gray /= max(float(np.linalg.norm(gray)), 1e-6)

        feature = np.concatenate([hist.astype(np.float32), gray], axis=0)
        feature /= max(float(np.linalg.norm(feature)), 1e-6)
        return feature.astype(np.float32)

    def _appearance_distance(self, feature_a, feature_b):
        if feature_a is None or feature_b is None:
            return 0.5
        similarity = float(np.clip(np.dot(feature_a, feature_b), -1.0, 1.0))
        return 1.0 - similarity

    def _archive_track(self, face_id):
        box, centroid = self.tracks[face_id]
        self.archived_tracks[face_id] = {
            "box": box,
            "centroid": centroid,
            "feature": self.track_features.get(face_id),
            "ttl": self.reid_ttl,
        }
        self.tracks.pop(face_id, None)
        self.disappeared.pop(face_id, None)
        self.track_features.pop(face_id, None)

    def _refresh_archived_tracks(self):
        expired_ids = []
        for face_id, state in self.archived_tracks.items():
            state["ttl"] -= 1
            if state["ttl"] <= 0:
                expired_ids.append(face_id)
        for face_id in expired_ids:
            self.archived_tracks.pop(face_id, None)

    def _try_reidentify(self, box, centroid, feature):
        best_id = None
        best_cost = np.inf
        for face_id, state in self.archived_tracks.items():
            dist = euclidean(state["centroid"], centroid)
            if dist > self.max_distance * 2.5:
                continue
            iou = self._iou(state["box"], box)
            appearance = self._appearance_distance(state["feature"], feature)
            cost = 0.65 * appearance + 0.25 * (dist / max(float(self.max_distance * 2.5), 1.0)) + 0.10 * (1.0 - iou)
            if cost < best_cost:
                best_cost = cost
                best_id = face_id

        if best_id is None or best_cost > 0.55:
            return None

        self.archived_tracks.pop(best_id, None)
        return best_id

    def update(self, boxes, frame=None):
        self._refresh_archived_tracks()

        # No detections this frame: age existing tracks.
        if not boxes:
            for face_id in list(self.tracks.keys()):
                self.disappeared[face_id] += 1
                if self.disappeared[face_id] > self.max_disappeared:
                    self._archive_track(face_id)
            return {}

        visible_assignments = {}
        detected_centroids = [self._centroid(box) for box in boxes]
        detected_features = (
            [self._extract_feature(frame, box) for box in boxes]
            if frame is not None
            else [None] * len(boxes)
        )
        existing_ids = list(self.tracks.keys())
        matched_det_indices = set()
        matched_ids = set()

        if existing_ids:
            # Global one-to-one assignment prevents ID swaps in multi-face scenes.
            cost = np.full((len(existing_ids), len(boxes)), np.inf, dtype=np.float32)
            for r, face_id in enumerate(existing_ids):
                prev_box, prev_centroid = self.tracks[face_id]
                prev_feature = self.track_features.get(face_id)
                for c, det_box in enumerate(boxes):
                    dist = euclidean(prev_centroid, detected_centroids[c])
                    if dist > self.max_distance:
                        continue
                    iou = self._iou(prev_box, det_box)
                    if iou < self.min_iou_for_match and dist > self.max_distance * 0.65:
                        continue
                    appearance = self._appearance_distance(prev_feature, detected_features[c])
                    dist_term = dist / max(float(self.max_distance), 1.0)
                    overlap_term = 1.0 - iou
                    cost[r, c] = (
                        (1.0 - self.appearance_weight) * (0.65 * dist_term + 0.35 * overlap_term)
                        + self.appearance_weight * appearance
                    )

            if np.isfinite(cost).any():
                try:
                    row_idx, col_idx = linear_sum_assignment(cost)
                except ValueError:
                    # Infeasible matrix (e.g., all-inf rows/cols) -> no matches this frame.
                    row_idx, col_idx = [], []

                for r, c in zip(row_idx, col_idx):
                    if not np.isfinite(cost[r, c]):
                        continue
                    face_id = existing_ids[r]
                    box = boxes[c]
                    centroid = detected_centroids[c]
                    if cost[r, c] > 0.85:
                        continue
                    visible_assignments[face_id] = box
                    self.tracks[face_id] = (box, centroid)
                    self.disappeared[face_id] = 0
                    self.track_features[face_id] = detected_features[c]
                    matched_det_indices.add(c)
                    matched_ids.add(face_id)

        # New or re-identified tracks for unmatched detections.
        for det_idx, box in enumerate(boxes):
            if det_idx in matched_det_indices:
                continue
            centroid = detected_centroids[det_idx]
            feature = detected_features[det_idx]
            face_id = self._try_reidentify(box, centroid, feature)
            if face_id is None:
                face_id = self.next_id
                self.next_id += 1
            visible_assignments[face_id] = box
            self.tracks[face_id] = (box, centroid)
            self.disappeared[face_id] = 0
            self.track_features[face_id] = feature
            matched_ids.add(face_id)

        # Age unmatched tracks and remove stale ones.
        for face_id in list(self.tracks.keys()):
            if face_id in matched_ids:
                continue
            self.disappeared[face_id] += 1
            if self.disappeared[face_id] > self.max_disappeared:
                self._archive_track(face_id)

        # Return only currently visible tracks.
        return visible_assignments
