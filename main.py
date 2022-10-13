import json
import os

from flasgger import LazyString, LazyJSONEncoder
from flasgger import Swagger
from flasgger.utils import swag_from
from flask import Flask, request, Response
from rembg import remove
from werkzeug.utils import secure_filename

from utils import *

ALLOWED_EXTENSIONS = ('png', 'jpg', 'jpeg', 'webp')

app = Flask(__name__)

UPLOAD_FOLDER = './uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

RESULT_FOLDER = "./results/"
if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)
app.config['RESULT_FOLDER'] = RESULT_FOLDER

app.config["SWAGGER"] = {"title": "Swagger-UI", "uiversion": 2}
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": "apispec_1",
            "route": "/apispec_1.json",
            "rule_filter": lambda rule: True,  # all in
            "model_filter": lambda tag: True,  # all in
        }
    ],
    "static_url_path": "/flasgger_static",
    # "static_folder": "static",  # must be set by user
    "swagger_ui": True,
    "specs_route": "/swagger/",
}
template = dict(
    swaggerUiPrefix=LazyString(lambda: request.environ.get("HTTP_X_SCRIPT_NAME", ""))
)
app.json_encoder = LazyJSONEncoder
swagger = Swagger(app, config=swagger_config, template=template)


@swag_from("./config_files/find_face_swagger_config.yml")
@app.route("/find-face/", methods = ['POST'])
def upload_file():
    """
    It takes the uploaded image, runs it through the face_landmark function, and then saves the image to
    the same directory as the app.py file
    :return: The path of the image that was uploaded.
    """
    try:
        file_loc, response_path = "", {}
        if request.method == "POST":
            if request.files['file'] != "":
                f = request.files['file']
                valid_input = f.filename.lower().endswith(ALLOWED_EXTENSIONS)
                if valid_input:
                    filename = secure_filename(f.filename)
                    file_loc = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    f.save(file_loc)
                    img = cv2.imread(file_loc)
                    img = face_landmark(img)
                    if img != "":
                        result_save = os.path.join(app.config['RESULT_FOLDER'], f.filename)
                        cv2.imwrite(result_save, img)
                        response_path["path"] = os.path.abspath(result_save)
                        os.remove(file_loc)
                        return Response(response_path["path"], status=200, mimetype='application/json')
                    else:
                        os.remove(file_loc)
                        return Response("Face not Found", status=204, mimetype='application/json')
                else:
                    return Response("Invalid File Format", status=422, mimetype='application/json')
    except Exception as e:
        return Response(e.args[0], status=500, mimetype='application/json')


@swag_from("./config_files/remove_background_swagger_config.yml")
@app.route("/rm-bg/", methods = ['POST'])
def remove_bg():
    """
    It takes the uploaded image, removes the background, and saves the image with the background removed
    :return: The path of the image with the background removed.
    """
    try:
        file_loc, response_path = "", {}
        if request.method == "POST":
            if request.files['file'] != "":
                f = request.files['file']
                valid_input = f.filename.lower().endswith(ALLOWED_EXTENSIONS)
                if valid_input:
                    filename = secure_filename(f.filename)
                    file_loc = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    f.save(file_loc)
                    img = cv2.imread(file_loc)
                    rem_bg = remove(img)
                    file_name = f.filename.split(".")
                    result_save = os.path.join(app.config['RESULT_FOLDER'], (file_name[0]+"-rm"+".png"))
                    cv2.imwrite(result_save, rem_bg)
                    response_path["file_path"] = os.path.abspath(result_save)
                    os.remove(file_loc)
                    return Response(response_path["file_path"], status=200, mimetype='application/json')
                else:
                    return Response("Invalid File Format", status=422, mimetype='application/json')
    except Exception as e:
        return Response(e.args[0], status=500, mimetype='application/json')



@swag_from("./config_files/compare_face_swagger_config.yml")
@app.route("/comapre-face/", methods=['POST'])
def compare_face_api():
    """
    It takes two images as input, compares them using the Azure Face API, and returns a JSON response
    with the confidence score
    :return: a response object with the result of the comparison.
    """
    try:
        if request.method == "POST":
            if request.files['file1'] != "" and request.files['file2'] != "":
                f1 = request.files['file1']
                f2 = request.files['file2']
                valid_input1, valid_input2 = f1.filename.lower().endswith(ALLOWED_EXTENSIONS), f2.filename.lower().endswith(ALLOWED_EXTENSIONS)
                if valid_input1 and valid_input2:
                    filename1, filename2 = secure_filename(f1.filename), secure_filename(f2.filename)
                    file_loc1, file_loc2 = os.path.join(app.config['UPLOAD_FOLDER'], filename1), os.path.join(app.config['UPLOAD_FOLDER'], filename2)
                    f1.save(file_loc1)
                    f2.save(file_loc2)

                    file_loc = os.listdir(app.config['UPLOAD_FOLDER'])

                    img1 = cv2.imread(file_loc1)
                    img2 = cv2.imread(file_loc2)
                    result =  json.dumps(compare_face(img1, img2))
                    for file in file_loc:
                        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))
                    return Response(result, status=200, mimetype='application/json')
                else:
                    return Response("Invalid File Format", status=422, mimetype='application/json')
            else:
                return Response("Both File required", status=405, mimetype='application/json')
    except Exception as e:
        return Response(e.args[0], status=500, mimetype='application/json')


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=False, port=5005)