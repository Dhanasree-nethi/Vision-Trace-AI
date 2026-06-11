import torch
import clip
import cv2

from PIL import Image


# Device
device = "cuda" if torch.cuda.is_available() else "cpu"


# Load CLIP model
clip_model, preprocess = clip.load(
    "ViT-B/32",
    device=device
)


# Reference features
reference_features = None


# Encode uploaded image
def load_reference_image(image_path):

    global reference_features

    try:

        image = Image.open(
            image_path
        ).convert("RGB")

        image_input = preprocess(
            image
        ).unsqueeze(0).to(device)

        with torch.no_grad():

            reference_features = clip_model.encode_image(
                image_input
            )

        print("REFERENCE IMAGE LOADED")

    except Exception as e:

        print("REFERENCE LOAD ERROR:", e)


# Compare webcam crop with reference image
def compare_object(cropped_frame):

    global reference_features

    if reference_features is None:
        return 0

    try:

        # Convert BGR to RGB
        rgb = cv2.cvtColor(
            cropped_frame,
            cv2.COLOR_BGR2RGB
        )

        pil_image = Image.fromarray(rgb)

        image_input = preprocess(
            pil_image
        ).unsqueeze(0).to(device)

        with torch.no_grad():

            frame_features = clip_model.encode_image(
                image_input
            )

        # Cosine similarity
        similarity = torch.cosine_similarity(
            reference_features,
            frame_features
        )

        score = similarity.item() * 100

        return score

    except Exception as e:

        print("COMPARE ERROR:", e)

        return 0