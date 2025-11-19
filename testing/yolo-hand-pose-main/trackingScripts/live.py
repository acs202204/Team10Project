import cv2
import torch
from ultralytics import YOLO

model = YOLO("/home/team10sp/Desktop/individualWork/Austin/YOLO/yolo_project/testing/yolo-hand-pose-main/model/best.pt")


# Move model to GPU if available
if torch.cuda.is_available():
    model.to("cuda")
    print("Running on GPU")
else:
    print("Running on CPU")


cap = cv2.VideoCapture(0)  # 0 = default webcam



while cap.isOpened():
    success, frame = cap.read()
    
    
    if success:

        results = model(frame)

        annotated_frame = results[0].plot()

        cv2.imshow("YOLOv8 Inference", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()