import cv2
import torch
import time
from ultralytics import YOLO

import logging
logging.getLogger('ultralytics').setLevel(logging.ERROR)

modelLoc = "/home/team10sp/Desktop/individualWork/Austin/YOLO/yolo_project/testing/yolo-hand-pose-main/model/best.pt" # finger Points Model
#modelLoc = "/home/team10sp/Desktop/individualWork/Austin/YOLO/yolo_project/testing/BaseModel/yolov8n.pt" # Basic Model
model = YOLO(modelLoc, verbose=False)

# Move model to GPU if available
if torch.cuda.is_available():
    model.to("cuda")
    device = "GPU: " + torch.cuda.get_device_name(0)
    print("Running on GPU")
else:
    print("Running on CPU")
    device = "CPU"

cap = cv2.VideoCapture(0)

# Set desired resolution and FPS (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# Variables for FPS calculation
prev_time = time.time()
fps = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Run YOLO inference
    results = model(frame, device=0, half=True)

    annotated = frame.copy()
    for r in results:
        if r.keypoints is not None:
            for kpts in r.keypoints.xy:
                for x, y in kpts:
                    cv2.circle(annotated, (int(x), int(y)), 3, (0,255,0), -1)


    # Get resolution from capture
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

     # Overlay FPS, resolution, and device info
    cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated, f"Resolution: {width}x{height}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated, f"Device: {device}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)


    # Show video
    cv2.imshow("Hand Tracking", annotated)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        hands = results[0].keypoints.xy
        if hands is not None and len(hands) > 0:
            for i, kpts in enumerate(hands):
                index_tip = kpts[8]
                x, y = index_tip[0].item(), index_tip[1].item()
                print(f"Hand {i+1} - Index tip: x={x:.2f}, y={y:.2f}")
        else:
            print("No hands detected.")

    # Exit on 'q' or ESC
    if key == ord("q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
