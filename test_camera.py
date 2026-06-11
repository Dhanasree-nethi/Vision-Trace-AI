import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not opening")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame not received")
        break

    cv2.imshow("Camera Test", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()