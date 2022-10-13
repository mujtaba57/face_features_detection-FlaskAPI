import cv2
from retinaface import RetinaFace
from deepface import DeepFace

def face_landmark(img):
    """
    It takes an image as input, and returns the image with the facial landmarks drawn on it

    :param img: The image to be processed
    :return: the image with the facial landmarks drawn on it.
    """
    result = RetinaFace.detect_faces(img)
    if isinstance(result, dict):
        for key in result.keys():
            identify = result[key]
            facial_area = identify["facial_area"]
            right_eye = identify["landmarks"]["right_eye"]
            left_eye = identify["landmarks"]["left_eye"]
            right_eye_cor = (int(right_eye[0]), int(right_eye[1]))
            left_eye_cor = (int(left_eye[0]), int(left_eye[1]))
            nose_cor = (int(identify["landmarks"]["nose"][0]), int(identify["landmarks"]["nose"][1]))
            mouth_right = (int(identify["landmarks"]["mouth_right"][0]), int(identify["landmarks"]["mouth_right"][1]))
            mouth_left = (int(identify["landmarks"]["mouth_left"][0]), int(identify["landmarks"]["mouth_left"][1]))
            prob_text = f"{round(identify['score'], 4)} % Find Coor"

            cv2.rectangle(img, (facial_area[0], facial_area[1]), (facial_area[2], facial_area[3]), (255, 255, 255), 7)
            cv2.circle(img, right_eye_cor, 20, (255, 0, 255), 5)
            cv2.circle(img, left_eye_cor, 20, (255, 0, 255), 5)
            cv2.circle(img, (nose_cor), 20, (0, 255, 0), 5)
            cv2.circle(img, mouth_right, 20, (0, 255, 255), 5)
            cv2.circle(img, mouth_left, 20, (0, 255, 255), 5)
            cv2.putText(img, prob_text, (facial_area[0] - 20, facial_area[3] + 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (150, 255, 0), 2, cv2.LINE_AA, False)
            return img
    else:
        return ""


def compare_face(img1, img2, model="VGG-Face", det_back="mtcnn"):
    """
    It takes two images as input and returns a dictionary of results
    
    :param img1: The first image to compare
    :param img2: The image to compare with img1
    :param model_name: The name of the model to be used, defaults to VGG-Face (optional)
    :param det_back: The detector backend to use. Can be either "mtcnn" or "ssd", defaults to mtcnn
    (optional)
    :return: The result is a dictionary with the following keys:
        - distance: The distance between the two faces. The lower the distance, the more similar the
    faces.
        - same_person: A boolean indicating whether the two faces belong to the same person.
        - face_match: A boolean indicating whether the two faces are a match.
        - model_name: The name
    """
    result = DeepFace.verify(img1, img2, model_name=model, detector_backend=det_back, prog_bar=True)
    return result

