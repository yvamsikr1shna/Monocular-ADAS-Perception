import cv2
import torch
import numpy as np
from ultralytics import YOLO

class ObjectDetector:
    """
    Module 2: Monocular ADAS Object Detector with ByteTRACK Multi-Object Tracking.
    Assigns persistent IDs across frames to eliminate spatial jitter during telemetry estimation.
    """
    def __init__(self, model_size: str = 'yolov8n.pt', confidence_threshold: float = 0.35):
        self.conf_thresh = confidence_threshold
        # Standard COCO vehicle and pedestrian classes
        self.target_classes = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        
        print(f"[INFO] Loading YOLOv8 Model with ByteTRACK ({model_size})...")
        self.model = YOLO(model_size)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def detect_and_track(self, frame: np.ndarray) -> list:
        """
        Executes YOLO inference with ByteTRACK object tracking.
        Retains persistent track IDs across frames.
        """
        results = self.model.track(
            frame, 
            conf=self.conf_thresh, 
            classes=list(self.target_classes.keys()), 
            device=self.device, 
            persist=True,       # Retain track state across consecutive frames
            tracker="bytetrack.yaml", # Built-in ByteTRACK configuration
            verbose=False
        )[0]

        detections = []
        if results.boxes is None or len(results.boxes) == 0:
            return detections

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = self.target_classes.get(cls_id, 'car')

            # Extract persistent tracking ID (default to -1 if unassigned in current frame)
            track_id = int(box.id[0].cpu().numpy()) if box.id is not None else -1

            # Midpoint of bottom bounding edge (tire-ground contact point)
            u_bottom = int((x1 + x2) / 2.0)
            v_bottom = int(y2)

            detections.append({
                'track_id': track_id,
                'bbox': [x1, y1, x2, y2],
                'contact_point': (u_bottom, v_bottom),
                'class_name': cls_name,
                'confidence': conf
            })

        return detections

    def render_clean_adas_hud(self, frame: np.ndarray, detections: list, geometry_mod, vp_y: int, mode: str = "2d_ground") -> np.ndarray:
        """
        Renders clean, production-style ADAS tracking overlays with persistent IDs.
        """
        vp_x = geometry_mod.cx

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            u_bot, v_bot = det['contact_point']
            cls_name = det['class_name']
            track_id = det['track_id']

            # Calculate metric depth Z (meters) using Module 1 pinhole model
            distance_z = geometry_mod.pixel_to_ground_distance(u_bot, v_bot, vp_y)

            if mode == "2d_ground":
                # --- OEM-Style Ground Target Tracking ---
                corner_len = min(14, int((x2 - x1) * 0.22))
                color_bracket = (0, 230, 120)

                # Top-Left Bracket
                cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color_bracket, 2)
                cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color_bracket, 2)
                # Top-Right Bracket
                cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color_bracket, 2)
                cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color_bracket, 2)
                # Bottom-Left Bracket
                cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color_bracket, 2)
                cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color_bracket, 2)
                # Bottom-Right Bracket
                cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color_bracket, 2)
                cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color_bracket, 2)

                # Ground Contact Projection Line (Road Anchor)
                cv2.line(frame, (x1, y2), (x2, y2), (0, 255, 200), 2)
                cv2.circle(frame, (u_bot, v_bot), 4, (0, 0, 255), -1)

                # Floating Metric Telemetry Label with Track ID
                id_str = f"ID:{track_id} " if track_id != -1 else ""
                label = f"{id_str}{cls_name.upper()} | {distance_z:.1f}m"
                cv2.putText(frame, label, (x1, max(y1 - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 150), 1, cv2.LINE_AA)

        return frame


if __name__ == "__main__":
    print("[TEST] Module 2 Detector with ByteTRACK initialized.")
