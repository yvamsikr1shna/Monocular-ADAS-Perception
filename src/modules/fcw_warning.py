import cv2
import numpy as np

class ForwardCollisionWarning:
    """
    Module 4: Ego-lane filtering, persistent-ID relative velocity tracking, 
    Time-to-Collision (TTC) calculations, and FCW safety alerts.
    """
    def __init__(self, fps: float = 30.0, ttc_critical: float = 1.8, ttc_warning: float = 3.0):
        self.fps = fps
        self.dt = 1.0 / fps
        self.ttc_critical = ttc_critical
        self.ttc_warning = ttc_warning
        
        # Track memory indexed by ByteTRACK persistent ID: {track_id: previous_z_meters}
        self.prev_distances = {}
        self.prev_velocities = {}

    def filter_ego_lane_targets(self, detections: list, lane_center_x: int, lane_width_px: int) -> tuple:
        """
        Identifies vehicles located inside the ego-lane path and picks the closest lead vehicle.
        """
        ego_targets = []
        half_lane = lane_width_px * 0.45

        for det in detections:
            u_bot, v_bot = det['contact_point']
            if abs(u_bot - lane_center_x) <= half_lane:
                det['is_ego_lane'] = True
                ego_targets.append(det)
            else:
                det['is_ego_lane'] = False

        lead_vehicle = None
        if ego_targets:
            # Sort by ground distance (bottom edge y-coordinate highest = closest)
            ego_targets.sort(key=lambda x: x['bbox'][3], reverse=True)
            lead_vehicle = ego_targets[0]

        return detections, lead_vehicle

    def process_fcw_telemetry(self, lead_vehicle: dict, geometry_mod, vp_y: int, frame_idx: int) -> dict:
        """
        Calculates metric distance, relative speed (m/s & km/h), TTC (seconds), and safety alert level.
        Uses persistent ByteTRACK ID to anchor delta calculations.
        """
        if lead_vehicle is None:
            return {'distance': 0.0, 'v_rel_ms': 0.0, 'v_rel_kmh': 0.0, 'ttc': float('inf'), 'alert': 'CLEAR'}

        u_bot, v_bot = lead_vehicle['contact_point']
        curr_z = geometry_mod.pixel_to_ground_distance(u_bot, v_bot, vp_y)
        track_id = lead_vehicle.get('track_id', -1)

        # Relative velocity calculation bound to persistent track ID
        if track_id != -1 and track_id in self.prev_distances:
            prev_z = self.prev_distances[track_id]
            raw_v_rel = (prev_z - curr_z) / self.dt  # Positive = Closing distance
            
            # Smooth velocity with exponential moving average filter
            prev_v = self.prev_velocities.get(track_id, raw_v_rel)
            v_rel = 0.75 * prev_v + 0.25 * raw_v_rel
        else:
            v_rel = 0.0

        if track_id != -1:
            self.prev_distances[track_id] = curr_z
            self.prev_velocities[track_id] = v_rel

        # Time-To-Collision (TTC)
        if v_rel > 0.5:  # Vehicle is closing in
            ttc = curr_z / v_rel
        else:            # Maintaining gap or separating
            ttc = float('inf')

        # Alert Logic
        if ttc <= self.ttc_critical and curr_z < 35.0:
            alert = "CRITICAL"
        elif ttc <= self.ttc_warning or curr_z < 15.0:
            alert = "WARNING"
        else:
            alert = "SAFE"

        return {
            'distance': curr_z,
            'v_rel_ms': v_rel,
            'v_rel_kmh': v_rel * 3.6,
            'ttc': ttc,
            'alert': alert
        }

    def render_fcw_hud(self, frame: np.ndarray, lead_vehicle: dict, fcw_data: dict) -> np.ndarray:
        """
        Renders lead vehicle highlights and FCW warning banners with persistent ID.
        """
        if lead_vehicle is None:
            return frame

        x1, y1, x2, y2 = lead_vehicle['bbox']
        alert = fcw_data['alert']
        ttc = fcw_data['ttc']
        z = fcw_data['distance']
        track_id = lead_vehicle.get('track_id', -1)

        id_str = f"ID:{track_id} | " if track_id != -1 else ""

        if alert == "CRITICAL":
            color = (0, 0, 255)
            banner_text = f"CRITICAL FCW | {id_str}TTC: {ttc:.1f}s | {z:.1f}m"
        elif alert == "WARNING":
            color = (0, 165, 255)
            banner_text = f"TAILGATING WARN | {id_str}TTC: {ttc:.1f}s | {z:.1f}m"
        else:
            color = (0, 255, 120)
            ttc_str = f"{ttc:.1f}s" if ttc != float('inf') else "SAFE"
            banner_text = f"LEAD VEHICLE | {id_str}TTC: {ttc_str} | {z:.1f}m"

        # Highlight Lead Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)

        # Warning Banner
        text_size, _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        bg_x1 = x1
        bg_y1 = max(y1 - 25, 20)
        bg_x2 = x1 + text_size[0] + 10
        bg_y2 = bg_y1 + text_size[1] + 8

        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
        cv2.putText(frame, banner_text, (bg_x1 + 5, bg_y2 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        return frame


if __name__ == "__main__":
    print("[TEST] Module 4 FCW with persistent tracking initialized.")
