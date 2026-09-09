import cv2
import numpy as np

class CameraGeometry:
    """
    Module 1: Estimates vanishing point (horizon) using edge & line analysis,
    calculates pinhole intrinsic matrices, and projects 2D pixels to 3D metric distances.
    """
    def __init__(self, image_width: int = 1280, image_height: int = 720, camera_height_m: float = 1.65):
        self.w = image_width
        self.h = image_height
        self.h_cam = camera_height_m

        # Estimate camera focal length (~1.2x width for standard automotive dashcams)
        self.fx = self.w * 1.2
        self.fy = self.fx
        self.cx = self.w / 2.0
        self.cy = self.h / 2.0

        # Pinhole Intrinsic Matrix K
        self.K = np.array([
            [self.fx, 0.0,     self.cx],
            [0.0,     self.fy, self.cy],
            [0.0,     0.0,     1.0]
        ], dtype=np.float32)

    def estimate_vanishing_point(self, frame: np.ndarray) -> tuple:
        """
        Uses Canny edges + Hough lines in the lower road region to compute 
        the horizon vanishing point intersection (vp_x, vp_y).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        # Region of Interest (ROI) covering ground lane area
        mask = np.zeros_like(edges)
        roi_pts = np.array([[
            (0, self.h),
            (int(self.w * 0.40), int(self.h * 0.50)),
            (int(self.w * 0.60), int(self.h * 0.50)),
            (self.w, self.h)
        ]], dtype=np.int32)
        cv2.fillPoly(mask, roi_pts, 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=40, minLineLength=40, maxLineGap=80)

        if lines is None:
            return int(self.cx), int(self.cy)

        left_slopes, left_intercepts = [], []
        right_slopes, right_intercepts = [], []

        for line in lines:
            coords = np.squeeze(line)
            if coords.ndim != 1 or len(coords) != 4:
                continue
            x1, y1, x2, y2 = coords
            if x2 == x1:
                continue
            
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1

            # Filter road line angles (left vs right lane lines)
            if -1.2 < slope < -0.3:
                left_slopes.append(slope)
                left_intercepts.append(intercept)
            elif 0.3 < slope < 1.2:
                right_slopes.append(slope)
                right_intercepts.append(intercept)

        if not left_slopes or not right_slopes:
            return int(self.cx), int(self.cy)

        m_left, c_left = np.mean(left_slopes), np.mean(left_intercepts)
        m_right, c_right = np.mean(right_slopes), np.mean(right_intercepts)

        if m_left == m_right:
            return int(self.cx), int(self.cy)

        # Intersection point x = (c2 - c1) / (m1 - m2)
        vp_x = (c_right - c_left) / (m_left - m_right)
        vp_y = m_left * vp_x + c_left

        # Constrain within physically plausible bounds on screen
        vp_x = int(np.clip(vp_x, self.w * 0.35, self.w * 0.65))
        vp_y = int(np.clip(vp_y, self.h * 0.40, self.h * 0.60))

        return vp_x, vp_y

    def pixel_to_ground_distance(self, u: float, v: float, vp_y: float) -> float:
        """
        Projects 2D image pixel (u, v) on the road plane to metric distance Z (meters).
        """
        delta_v = v - vp_y
        if delta_v <= 1.0:
            return 100.0  # Above or at horizon -> far field limit

        # Metric depth Z = (fy * H_cam) / delta_v
        Z = (self.fy * self.h_cam) / delta_v
        return float(np.clip(Z, 1.0, 120.0))

    def render_subtle_geometry_overlay(self, frame: np.ndarray, vp_x: int, vp_y: int) -> np.ndarray:
        """
        Renders clean, non-intrusive horizon calibration graphics.
        """
        # Subtle horizontal horizon line (Cyan, thin)
        cv2.line(frame, (0, vp_y), (self.w, vp_y), (255, 200, 0), 1, cv2.LINE_AA)
        
        # Vanishing Point Reticle
        cv2.drawMarker(frame, (vp_x, vp_y), (255, 200, 0), cv2.MARKER_CROSS, 12, 1, cv2.LINE_AA)
        
        # Subtle label
        cv2.putText(frame, f"HORIZON VP: ({vp_x}, {vp_y})", (vp_x + 15, vp_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 0), 1, cv2.LINE_AA)

        return frame


if __name__ == "__main__":
    print("[TEST] Running Module 1 verification...")
    geom = CameraGeometry()
    test_dist = geom.pixel_to_ground_distance(u=640, v=680, vp_y=380)
    print(f"[TEST] Pixel at y=680 mapped to ground depth: {test_dist:.2f} meters")
