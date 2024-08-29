# /// script
# dependencies = [
#   "opencv-python",
#   "numpy",
#   "tqdm",
#   "dlib",
#   "imutils",
# ]
# ///

import os
from pathlib import Path

import cv2
import dlib
import numpy as np
from imutils import face_utils
from tqdm import tqdm

FACES_DIR = Path("./faces")


def download_dat():
    url = "https://github.com/GuoQuanhao/68_points/raw/master/shape_predictor_68_face_landmarks.dat"
    dat_file = "shape_predictor_68_face_landmarks.dat"

    if not os.path.exists(dat_file):
        print(f"Downloading {dat_file}...")
        import urllib.request

        urllib.request.urlretrieve(url, dat_file)
        print("Download complete.")

    return dat_file


def detect_eyes(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)
        left_eye = shape[36:42].mean(axis=0).astype("int")
        right_eye = shape[42:48].mean(axis=0).astype("int")
        return [tuple(left_eye), tuple(right_eye)]
    return None


def process_images():
    aligned_images = []
    for img_path in tqdm(
        list(FACES_DIR.glob("*.[jJ][pP][gG]"))
        + list(FACES_DIR.glob("*.[jJ][pP][eE][gG]"))
        + list(FACES_DIR.glob("*.[pP][nN][gG]"))
    ):
        # Read the image file
        img = cv2.imread(str(img_path))
        # Detect eyes in the image
        eyes = detect_eyes(img)
        if eyes:
            left_eye, right_eye = eyes
            # desired eye positions (as a fraction of image width and height)
            desired_left_eye = (0.45, 0.35)
            desired_right_eye = (0.55, 0.35)

            desired_width = 1024
            desired_height = 1024

            # calc the angle between the eyes
            dY = right_eye[1] - left_eye[1]
            dX = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dY, dX))

            # calc scale factor to resize the image
            dist = np.sqrt((dX**2) + (dY**2))
            desired_dist = desired_right_eye[0] - desired_left_eye[0]
            desired_dist *= desired_width
            scale = desired_dist / dist

            eyes_center = (
                (left_eye[0] + right_eye[0]) // 2,
                (left_eye[1] + right_eye[1]) // 2,
            )
            eyes_center = tuple(map(float, eyes_center))

            # rotation matrix to align the face
            M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

            # Update the translation component of the transformation matrix
            # This step is crucial for positioning the aligned face correctly in the output image
            # calc the desired center of the left eye in the output image
            tX = desired_width * 0.5  # Center horizontally
            # position vertically based on desired eye position
            tY = desired_height * desired_left_eye[1]

            # adjust the translation to move the actual eye center to the desired position
            # ensures that the eyes are in the correct location after transformation
            M[0, 2] += tX - eyes_center[0]
            M[1, 2] += tY - eyes_center[1]

            # apply the affine trans
            output = cv2.warpAffine(
                img, M, (desired_width, desired_height), flags=cv2.INTER_CUBIC
            )

            aligned_images.append(output)
        else:
            print(f"No eyes detected in image: {img_path}")
    return aligned_images


def create_video(images, output_path="output.mp4", fps=24):
    height, width = images[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img in images:
        out.write(img)

    out.release()


if __name__ == "__main__":
    download_dat()
    aligned_images = process_images()
    if aligned_images:
        create_video(aligned_images)
        print("Video created successfully!")
    else:
        print(
            "No images were processed. Check if there are .jpg files in the FACES_DIR."
        )
