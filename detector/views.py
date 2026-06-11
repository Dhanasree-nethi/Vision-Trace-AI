from django.shortcuts import render, redirect
from django.http import StreamingHttpResponse
from django.core.files.storage import FileSystemStorage

from ultralytics import YOLO

from .object_matcher import (
    load_reference_image,
    compare_object
)

import cv2
import pyttsx3
import threading
import time


# Load YOLO
model = YOLO("yolov8n.pt")


# Voice engine
engine = pyttsx3.init()


# Globals
target_object = None
last_spoken = ""


# Webcam
camera = cv2.VideoCapture(0)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# Voice alert
def speak(text):

    engine.say(text)

    engine.runAndWait()


# Upload page
def upload_image(request):

    global target_object
    global last_spoken

    if request.method == "POST":

        image = request.FILES.get("image")

        target_object = request.POST.get(
            "object_name"
        )

        if target_object:

            target_object = target_object.lower().strip()

        # Save uploaded image
        if image:

            fs = FileSystemStorage(location="media")

            saved_name = fs.save(
                image.name,
                image
            )

            image_path = fs.path(saved_name)

            # Load CLIP reference image
            load_reference_image(image_path)

        last_spoken = ""

        print("SEARCHING FOR:", target_object)

        return redirect("/ai-camera/")

    return render(
        request,
        "upload.html"
    )


# AI Camera Page
def ai_camera(request):

    return render(
        request,
        "aicamera.html"
    )


# Generate video frames
def generate_frames():

    global target_object
    global last_spoken

    while True:

        success, frame = camera.read()

        if not success:
            break

        # Run YOLO
        results = model(
            frame,
            verbose=False
        )

        boxes = results[0].boxes

        names = model.names

        found = False

        for box in boxes:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            cls = int(box.cls[0])

            conf = float(box.conf[0])

            confidence = int(conf * 100)

            label = names[cls].lower().strip()

            # Default box
            color = (0, 0, 255)

            text = f"{label} {confidence}%"

            # Synonyms
            synonyms = {
                "phone": "cell phone",
                "mobile": "cell phone",
                "bag": "backpack"
            }

            user_object = target_object

            if user_object in synonyms:

                user_object = synonyms[user_object]

            # Object class matched
            if user_object and user_object == label:

                # Crop detected object
                cropped = frame[
                    y1:y2,
                    x1:x2
                ]

                # Check crop exists
                if cropped.size != 0:

                    # Compare using CLIP
                    similarity = compare_object(
                        cropped
                    )

                    print(
                        "SIMILARITY:",
                        similarity
                    )

                    # Threshold
                    if similarity > 75:

                        found = True

                        color = (0, 255, 0)

                        text = (
                            f"{label} FOUND "
                            f"{similarity:.1f}%"
                        )

                        # Voice alert once
                        if last_spoken != user_object:

                            threading.Thread(
                                target=speak,
                                args=(
                                    f"{user_object} found",
                                )
                            ).start()

                            last_spoken = user_object

                    else:

                        text = (
                            f"{label} "
                            f"{similarity:.1f}%"
                        )

            # Draw rectangle
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                3
            )

            # Draw label
            cv2.putText(
                frame,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        # Top status
        if found:

            cv2.putText(
                frame,
                "OBJECT FOUND!",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                4
            )

        elif target_object:

            cv2.putText(
                frame,
                f"Searching for {target_object}...",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                3
            )

        # Convert frame
        ret, buffer = cv2.imencode(
            ".jpg",
            frame
        )

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame +
            b'\r\n'
        )

        time.sleep(0.03)


# Video feed
def video_feed(request):

    return StreamingHttpResponse(
        generate_frames(),
        content_type=(
            'multipart/x-mixed-replace; boundary=frame'
        )
    )