from ultralytics import YOLO
import cv2

# Load YOLOv8 model (you can fine-tune it on hands later)
model = YOLO("yolov8n.pt")  # small model for real-time speed

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    # Visualize
    annotated_frame = results[0].plot()
    cv2.imshow("Hand Detection", annotated_frame)

    if cv2.waitKey(1) == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
