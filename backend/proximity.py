import cv2
from ultralytics import YOLO

model = YOLO("yolov8m.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open video stream")
    exit()

ALERT_CLASSES = {"person", "car", "truck", "bus", "bicycle", "motorcycle"}


def proximity_level(box_area, frame_area):
    ratio = box_area / frame_area

    if ratio > 0.20:
        return "COLLISION IMMINENT", (0, 0, 255)
    elif ratio > 0.10:
        return "VERY CLOSE", (0, 100, 255)
    elif ratio > 0.04:
        return "CLOSE", (0, 255, 255)
    else:
        return None, None


while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    frame_area = w * h

    results = model.predict(frame, conf=0.25, iou=0.5, verbose=False)
    annotated_frame = results[0].plot()

    names = model.names
    highest_warning = None
    highest_color = None

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = names[cls_id]

        if label not in ALERT_CLASSES:
            continue

        x1, y1, x2, y2 = box.xyxy[0]
        box_area = (x2 - x1) * (y2 - y1)

        level, color = proximity_level(box_area, frame_area)

        if level:
            highest_warning = level
            highest_color = color

    if highest_warning:
        cv2.putText(
            annotated_frame,
            highest_warning,
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            highest_color,
            3
        )

    cv2.imshow("YOLO Live Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()