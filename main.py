import sys
import os
import cv2

# Append project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from modules.video_loader import VideoStreamHandler
from modules.camera_geometry import CameraGeometry
from modules.object_detector import ObjectDetector
from modules.lane_detector import LaneDetector
from modules.fcw_warning import ForwardCollisionWarning

def run_full_adas_stack():
    input_video = "data/input/sample_dashcam.mp4"
    output_video = "data/output/m0_to_m4_full_adas_output.mp4"

    print("[INFO] Initializing Full Monocular ADAS Perception Stack (M0 -> M4)...")
    
    stream = VideoStreamHandler(video_path=input_video, output_path=output_video)
    geom = CameraGeometry(image_width=stream.target_w, image_height=stream.target_h)
    detector = ObjectDetector(confidence_threshold=0.35)
    lane_mod = LaneDetector(image_width=stream.target_w, image_height=stream.target_h)
    fcw = ForwardCollisionWarning(fps=30.0, ttc_critical=1.8, ttc_warning=3.0)

    frame_idx = 0
    while True:
        ret, frame = stream.read_frame()
        if not ret:
            break

        frame_idx += 1

        # 1. Horizon Calibration
        vp_x, vp_y = geom.estimate_vanishing_point(frame)

        # 2. Lane Segmentation
        lane_overlay, curve_rad, offset = lane_mod.process_lane_geometry(frame)
        annotated_frame = cv2.addWeighted(frame, 1.0, lane_overlay, 0.25, 0)

        # 3. Object Detection
        detections = detector.detect_and_track(frame)

        # 4. Ego Lane Target Filtering & FCW Analysis
        lane_center_x = geom.cx  # Approximate camera centerline
        lane_width_px = int(stream.target_w * 0.35)
        
        detections, lead_vehicle = fcw.filter_ego_lane_targets(detections, lane_center_x, lane_width_px)
        fcw_data = fcw.process_fcw_telemetry(lead_vehicle, geom, vp_y, frame_idx)

        # 5. Render Detection Overlays & FCW Alerts
        annotated_frame = detector.render_clean_adas_hud(
            annotated_frame, detections, geom, vp_y, mode="2d_ground"
        )
        annotated_frame = fcw.render_fcw_hud(annotated_frame, lead_vehicle, fcw_data)

        # 6. Top Telemetry Banner
        annotated_frame = geom.render_subtle_geometry_overlay(annotated_frame, vp_x, vp_y)
        
        offset_dir = "RIGHT" if offset > 0 else "LEFT"
        hud_text = f"CURVE: {curve_rad:.0f}m | OFFSET: {abs(offset):.2f}m {offset_dir} | FCW: {fcw_data['alert']}"
        annotated_frame = stream.draw_subtle_hud(annotated_frame, frame_idx, status_text=hud_text)

        stream.write_frame(annotated_frame)

        if frame_idx % 100 == 0:
            print(f"[PROGRESS] Processed {frame_idx}/{stream.total_frames} frames...")

    stream.release()
    print(f"[SUCCESS] Complete ADAS output saved to: {output_video}")

if __name__ == "__main__":
    run_full_adas_stack()
