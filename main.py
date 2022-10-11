import os
from rembg import remove
import cv2
from flask import Flask, request
from retinaface import RetinaFace
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = './uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER




def face_landmark(img):
    """
    It takes an image as input, and returns the image with the facial landmarks drawn on it
    
    :param img: The image to be processed
    :return: the image with the facial landmarks drawn on it.
    """
    result = RetinaFace.detect_faces(img)
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

        cv2.rectangle(img, (facial_area[0], facial_area[1]), (facial_area[2], facial_area[3]), (255,255, 255), 10)
        cv2.circle(img, right_eye_cor, 20, (255,0, 255), 10)
        cv2.circle(img, left_eye_cor, 20, (255,0, 255), 10)
        cv2.circle(img, (nose_cor), 20, (0, 255, 0), 10)
        cv2.circle(img, mouth_right, 20, (0,255, 255), 10)
        cv2.circle(img, mouth_left, 20, (0,255, 255), 10)
        cv2.putText(img, prob_text, (facial_area[0]-20, facial_area[3]+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 255, 0), 2, cv2.LINE_AA, False)
        return img




@app.route("/find-face/", methods = ['POST'])
def upload_file():
    """
    It takes the uploaded image, runs it through the face_landmark function, and then saves the image to
    the same directory as the app.py file
    :return: The path of the image that was uploaded.
    """
    global file_loc, response_path
    if request.method == "POST":
        if request.files['file'] != "":
            f = request.files['file']
            filename = secure_filename(f.filename)
            file_loc = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(file_loc)

            img = cv2.imread(file_loc)
            img = face_landmark(img)
            cv2.imwrite(f.filename, img)
            response_path = os.path.abspath(f.filename)
             
        os.remove(file_loc)  
        return response_path


@app.route("/rm-bg/", methods = ['POST'])
def remove_bg():
    """
    It takes the uploaded image, removes the background, and saves the image with the background removed
    :return: The path of the image with the background removed.
    """
    global file_loc, response_path
    if request.method == "POST":
        if request.files['file'] != "":
            f = request.files['file']
            filename = secure_filename(f.filename)
            file_loc = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(file_loc)

            img = cv2.imread(file_loc)
            rem_bg = remove(img)
            file_name = f.filename.split(".")
            cv2.imwrite(file_name[0]+"-rm"+".png", rem_bg)
            response_path = os.path.abspath(os.path.join(file_name[0]+"-rm"+".png"))

        os.remove(file_loc)
        return response_path

@app.route("/comapre-face/", methods=['POST'])
def compare_face():
    if request.method == "POST":
        files = request.files.getlist('file')
        file_names = []
        for file in files:
            if file:
                filename = secure_filename(file.filename)
                file_names.append(filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                print(filename)
        return "both file recieved"
    else:
        return "error"

if __name__ == '__main__':
   app.run(host="0.0.0.0", debug=False, port=5000)