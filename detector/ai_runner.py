import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def run_ai():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return "Camera Error"

    found = False

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        results = model(frame)

        annotated = results[0].plot()

        for box in results[0].boxes:

            conf = float(box.conf[0]) * 100

            if conf > 80:
                found = True

        if found:
            cv2.putText(
                annotated,
                "OBJECT FOUND",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

            cv2.imshow("VisionTrace AI", annotated)

            cv2.waitKey(2000)
            break

        cv2.imshow("VisionTrace AI", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return "Completed"