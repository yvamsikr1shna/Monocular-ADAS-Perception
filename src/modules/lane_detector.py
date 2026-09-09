import cv2
import numpy as np

class LaneDetector:
    """
    Module 3: Handles Bird's-Eye View perspective warping, sliding window 
    polynomial curve fitting, lateral ego-offset, and road curvature radius.
    """
    def __init__(self, image_width: int = 1280, image_height: int = 720):
        self.w = image_width
        self.h = image_height
        
        # Metric conversion factors
        self.ym_per_pix = 30.0 / 720  
        self.xm_per_pix = 3.7 / 650   

        # Source perspective coordinates (trapezoid on front camera view)
        src_pts = np.float32([
            [int(self.w * 0.43), int(self.h * 0.55)],
            [int(self.w * 0.57), int(self.h * 0.55)],
            [int(self.w * 0.90), self.h],
            [int(self.w * 0.10), self.h]
        ])
        
        # Destination top-down coordinates
        dst_pts = np.float32([
            [int(self.w * 0.20), 0],
            [int(self.w * 0.80), 0],
            [int(self.w * 0.80), self.h],
            [int(self.w * 0.20), self.h]
        ])
        
        self.M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self.M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)

    def process_lane_geometry(self, frame: np.ndarray) -> tuple:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv, (15, 80, 100), (35, 255, 255))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        white_mask = cv2.inRange(gray, 190, 255)
        combined_binary = cv2.bitwise_or(yellow_mask, white_mask)

        warped = cv2.warpPerspective(combined_binary, self.M, (self.w, self.h), flags=cv2.INTER_LINEAR)

        histogram = np.sum(warped[self.h // 2:, :], axis=0)
        midpoint = int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        nwindows = 9
        window_height = int(self.h // nwindows)
        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current, rightx_current = leftx_base, rightx_base
        margin, minpix = 80, 40
        left_lane_inds, right_lane_inds = [], []

        for window in range(nwindows):
            win_y_low = self.h - (window + 1) * window_height
            win_y_high = self.h - window * window_height

            good_left = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= leftx_current - margin) & (nonzerox < leftx_current + margin)).nonzero()[0]
            good_right = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= rightx_current - margin) & (nonzerox < rightx_current + margin)).nonzero()[0]

            left_lane_inds.append(good_left)
            right_lane_inds.append(good_right)

            if len(good_left) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left]))
            if len(good_right) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right]))

        left_lane_inds = np.concatenate(left_lane_inds) if len(left_lane_inds) > 0 else np.array([])
        right_lane_inds = np.concatenate(right_lane_inds) if len(right_lane_inds) > 0 else np.array([])

        ploty = np.linspace(0, self.h - 1, self.h)
        lane_overlay = np.zeros_like(frame)

        if len(left_lane_inds) > 50 and len(right_lane_inds) > 50:
            leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
            rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]

            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)

            left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

            left_fit_cr = np.polyfit(lefty * self.ym_per_pix, leftx * self.xm_per_pix, 2)
            y_eval = np.max(ploty) * self.ym_per_pix
            curverad = ((1 + (2 * left_fit_cr[0] * y_eval + left_fit_cr[1])**2)**1.5) / np.absolute(2 * left_fit_cr[0])

            car_pos = self.w / 2.0
            lane_center = (left_fitx[-1] + right_fitx[-1]) / 2.0
            center_offset = (car_pos - lane_center) * self.xm_per_pix

            warp_zero = np.zeros_like(warped).astype(np.uint8)
            color_warp = cv2.merge((warp_zero, warp_zero, warp_zero))

            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))

            cv2.fillPoly(color_warp, np.int_([pts]), (0, 180, 80))
            lane_overlay = cv2.warpPerspective(color_warp, self.M_inv, (self.w, self.h))

            return lane_overlay, curverad, center_offset

        return lane_overlay, 0.0, 0.0
