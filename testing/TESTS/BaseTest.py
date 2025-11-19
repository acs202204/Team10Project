from ultralytics import YOLO

model = YOLO("venv/testing/yolo-hand-pose-main/model/best.pt")  # nano model
results = model("testImage.jpg")  # replace with your image
results.show()
