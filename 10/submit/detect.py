from ultralytics import YOLO



#  img = cv2.imread("walkers.jpg")

model = YOLO('yolov8n.pt')
model.predict(
        source='walkers.jpg',
        conf = 0.24,
        save = True,
        show = True
        )


