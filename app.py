from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

app = Flask(__name__)

CSV_PATH = "iris_atm_dataset.csv"
DATASET_PATH = Path("dataset")

data = pd.read_csv(CSV_PATH)
data.columns = data.columns.str.strip()

sift = cv2.SIFT_create(nfeatures=3000)
matcher = cv2.BFMatcher()


def get_features(image):
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (640, 480))

    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return descriptors


def compare_images(query, registered):
    q = get_features(query)
    r = get_features(registered)

    if q is None or r is None:
        return 0

    if len(q) < 2 or len(r) < 2:
        return 0

    try:
        matches = matcher.knnMatch(q, r, k=2)
    except cv2.error:
        return 0

    good = []

    for pair in matches:
        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < 0.75 * n.distance:
            good.append(m)

    if not good:
        return 0

    return len(good) / min(len(q), len(r))


def find_image(student_id, image_file):
    path = DATASET_PATH / student_id / image_file

    if path.exists():
        return path

    return None


def recognize_iris(query_image):
    best_score = 0
    best_row = None

    for _, row in data.iterrows():
        student_id = str(row["student_id"]).strip()
        image_file = str(row["image_file"]).strip()

        image_path = find_image(student_id, image_file)

        if image_path is None:
            continue

        registered = cv2.imread(str(image_path))

        if registered is None:
            continue

        score = compare_images(query_image, registered)

        if score > best_score:
            best_score = score
            best_row = row

    threshold = 0.04

    if best_row is None or best_score < threshold:
        return None, best_score

    details = {}

    for column in data.columns:
        value = best_row[column]

        if pd.isna(value):
            value = ""

        details[column] = str(value)

    return details, best_score


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/recognize", methods=["POST"])
def recognize():
    if "image" not in request.files:
        return jsonify({
            "success": False,
            "message": "No image received.",
            "score": 0
        })

    file = request.files["image"]

    image_data = file.read()
    image_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({
            "success": False,
            "message": "Invalid image.",
            "score": 0
        })

    customer, score = recognize_iris(image)

    percentage = round(score * 100, 2)

    if customer is None:
        return jsonify({
            "success": False,
            "message": "CUSTOMER DATA NOT FOUND",
            "score": percentage
        })

    return jsonify({
        "success": True,
        "message": "IRIS MATCH FOUND",
        "score": percentage,
        "customer": customer
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
