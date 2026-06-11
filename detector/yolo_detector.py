import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera not opening")
    exit()

print("VisionTrace AI Running... Press Q to exit")

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Frame not received")
        break

    # Run YOLO detection
    results = model(frame)

    annotated_frame = results[0].plot()

    found = False

    # Check detections safely
    if results[0].boxes is not None:

        for box in results[0].boxes:

            conf = float(box.conf[0]) * 100
            print(f"Confidence: {conf:.2f}%")

            # SAFE threshold (YOLO default confidence is already strong)
            if conf > 80:
                found = True

    if found:
        cv2.putText(
            annotated_frame,
            "OBJECT DETECTED",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    cv2.imshow("VisionTrace AI", annotated_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()