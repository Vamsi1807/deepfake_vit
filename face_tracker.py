from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import euclidean


class FaceTracker:
    def __init__(self, max_distance=60, max_disappeared=30):
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared
        self.next_id = 0
        self.tracks = {}  # id -> (box, centroid)
        self.disappeared = defaultdict(int)

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

    def update(self, boxes):
        # No detections this frame: age existing tracks.
        if not boxes:
            for face_id in list(self.tracks.keys()):
                self.disappeared[face_id] += 1
                if self.disappeared[face_id] > self.max_disappeared:
                    self.tracks.pop(face_id, None)
                    self.disappeared.pop(face_id, None)
            return {}

        visible_assignments = {}
        detected_centroids = [self._centroid(box) for box in boxes]
        existing_ids = list(self.tracks.keys())
        matched_det_indices = set()

        if existing_ids:
            # Global one-to-one assignment prevents ID swaps in multi-face scenes.
            cost = np.full((len(existing_ids), len(boxes)), np.inf, dtype=np.float32)
            for r, face_id in enumerate(existing_ids):
                prev_box, prev_centroid = self.tracks[face_id]
                for c, det_box in enumerate(boxes):
                    dist = euclidean(prev_centroid, detected_centroids[c])
                    if dist > self.max_distance:
                        continue
                    iou = self._iou(prev_box, det_box)
                    dist_term = dist / max(float(self.max_distance), 1.0)
                    overlap_term = 1.0 - iou
                    cost[r, c] = 0.7 * dist_term + 0.3 * overlap_term

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
                    visible_assignments[face_id] = box
                    self.tracks[face_id] = (box, centroid)
                    self.disappeared[face_id] = 0
                    matched_det_indices.add(c)

        # New tracks for unmatched detections.
        for det_idx, box in enumerate(boxes):
            if det_idx in matched_det_indices:
                continue
            face_id = self.next_id
            self.next_id += 1
            centroid = detected_centroids[det_idx]
            visible_assignments[face_id] = box
            self.tracks[face_id] = (box, centroid)
            self.disappeared[face_id] = 0

        # Age unmatched tracks and remove stale ones.
        for face_id in list(self.tracks.keys()):
            if face_id in visible_assignments:
                continue
            self.disappeared[face_id] += 1
            if self.disappeared[face_id] > self.max_disappeared:
                self.tracks.pop(face_id, None)
                self.disappeared.pop(face_id, None)

        # Return only currently visible tracks.
        return visible_assignments
