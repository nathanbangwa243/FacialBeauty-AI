import cv2
import numpy as np
import imutils
import dlib
import sys
import os


# CONFIG
OPENCV_FKP_MODEL = os.path.join(
    os.getcwd(), "models", "shape_predictor_68_face_landmarks.dat")


def renderFace2(image, landmarks, color=(0, 255, 0), radius=3):
    """Draw Facial landmarks    

    Args:
        image (np.array): L'image
        landmarks (np.array): facial keypoints
        color (tuple, optional): Facial keypoints color. Defaults to (0, 255, 0).
        radius (int, optional): FKP Radius. Defaults to 3.
    """

    # make image copy
    imageCopy = np.array(image)

    for p in landmarks.parts():
        cv2.circle(imageCopy, (p.x, p.y), radius, color, -1)

    return imageCopy


def renderFKPs(landmarks) -> dict:
    """Extract facial keypoints

    Args:
        landmarks ([type]): [description]

    Returns:
        dict: {fkp: [x, y]}
    """

    keypoints = {}

    index = 0

    for fkp in landmarks:
        keypoints[index] = [fkp.x, fkp.y]

        index += 1

    return keypoints


def makeInference(image: np.array) -> list:
    """Make prediction

    Args:
        image (np.array): Preprocess image matrix

    Returns:
        dict: facial keypoints
        [
            {
                "faceLoc": {
                    "left": face.left(),
                    "top": face.top(),
                    "right": face.right(),
                    "bottom": face.bottom(),
                }

                "keypoints":{
                    # fkp : [x,y]
                }
            }
        ]
    """

    logging.info(f"[makeInference] Start")
    logging.info(f"[makeInference] Image shape : {image.shape}")

    # face detector
    detector = dlib.get_frontal_face_detector()
    # 68 points predictor
    predictor = dlib.shape_predictor(OPENCV_FKP_MODEL)

    # copy image
    imageCopy = np.copy(image)

    # Resize the frame
    imageCopy = imutils.resize(imageCopy, width=640)
    imageCopy = imutils.resize(imageCopy, height=480)

    # Convert to RGB
    grayImage = cv2.cvtColor(imageCopy, cv2.COLOR_BGR2RGB)

    # response
    response = []

    # Detect faces in the frame
    faces = detector(grayImage, 0)

    # Iterate over faces in the frame
    for face in faces:
        newRect = dlib.rectangle(
            int(face.left()),   # X1
            int(face.top()),    # Y1
            int(face.right()),  # X2
            int(face.bottom())  # Y2
        )

        # Find face landmarks by providing reactangle for each face
        shape = predictor(grayImage, newRect)

        # Draw facial landmarks
        renderFace2(imageCopy, shape)

        # render keypoints
        keypoints = renderFKPs(shape)

        # face localization
        faceLoc = {
            "left": face.left(),
            "top": face.top(),
            "right": face.right(),
            "bottom": face.bottom(),
        }

        # add faces infos
        response.append(
            {
                "faceLocation": faceLoc,
                "keypoints": keypoints
            }
        )

    return response
