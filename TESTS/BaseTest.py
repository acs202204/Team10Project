from ultralytics import YOLO

model = YOLO("/home/team10sp/Desktop/individualWork/Austin/YOLO/yolo_project/testing/yolo-hand-pose-main/model/best.pt")  # nano model
results = model("testImage.jpg")  # replace with your image
results.show()
