# Monocular ADAS Perception Pipeline 

A modular, real-time Monocular Advanced Driver Assistance System (ADAS) perception pipeline engineered in Python, OpenCV, PyTorch, and YOLOv8 with ByteTRACK.

The pipeline estimates 3D metric spatial geometry, tracks lane boundaries, monitors surrounding vehicle kinematics, and generates Time-To-Collision (TTC) Forward Collision Warnings (FCW) from a standard monocular dashcam stream.

---

## Architecture & Core Modules

* **Module 0: Stream Handler & Performance Engine (`video_loader.py`)**
  * Manages video frame decoding, targeted spatial scaling, and frame-rate synchronization.
  * Renders an OEM-style status banner displaying system health, tracking frame count, ego offset, and lane curvature radius.

* **Module 1: Dynamic Camera Geometry & Ground Projection (`camera_geometry.py`)**
  * Dynamically estimates the optical horizon and vanishing point ($v_x, v_y$) across frames.
  * Maps 2D pixel coordinates $(u, v)$ from the vehicle ground contact point to 3D metric ground distances ($Z\text{-meters}$) using a pinhole camera calibration model.

* **Module 2: Vehicle Detection & Multi-Object Tracking (`object_detector.py`)**
  * Detects surrounding vehicles (cars, trucks, buses, motorcycles) and pedestrians using YOLOv8.
  * Integrates ByteTRACK multi-object tracking to assign persistent tracking IDs across frames, removing spatial jitter and stabilizing relative speed calculations.
  * Draws ground-anchored corner brackets and road projection markers beneath detected vehicles for clean visual telemetry.

* **Module 3: Perspective Lane Segmentation & Curvature Fitting (`lane_detector.py`)**
  * Applies Bird’s-Eye View (BEV) perspective warping to project the road into an overhead plane.
  * Tracks lane lines using 2nd-degree polynomial curve fitting ($y = ax^2 + bx + c$).
  * Calculates real-time lane curvature radius ($R\text{-meters}$) and lateral vehicle offset relative to lane center.

* **Module 4: Forward Collision Warning & Risk Assessment (`fcw_warning.py`)**
  * Isolates detected vehicles located directly in the ego vehicle's driving path.
  * Computes relative closing speed ($m/s$ and $km/h$) using persistent track ID delta tracking over time.
  * Determines collision time horizon ($TTC = \frac{Z}{v_{rel}}$).
  * Triggers dynamic HUD safety alerts:
    * **SAFE (Green):** Stable distance or separating velocity.
    * **TAILGATING WARN (Amber):** Close proximity or low TTC ($< 3.0s$).
    * **CRITICAL FCW (Red):** High-risk closing speed or imminent collision threshold ($< 1.8s$).

---

## Tech Stack

* **Language:** Python 3
* **Computer Vision:** OpenCV, NumPy
* **Deep Learning & Detection:** PyTorch, Ultralytics YOLOv8
* **Multi-Object Tracking:** ByteTRACK

---
