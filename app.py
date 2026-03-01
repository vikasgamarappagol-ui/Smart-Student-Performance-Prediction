from flask import Flask, render_template, request, redirect
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import r2_score, accuracy_score
import os

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"

# Create uploads folder if not exists
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# -----------------------------
# Load Models
# -----------------------------
lr = joblib.load("linear_model.pkl")
dt = joblib.load("decision_tree.pkl")
knn = joblib.load("knn_model.pkl")
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

# Default dataset
df = pd.read_csv("student_synthetic_data_final.csv")

# ---------------- HOME ----------------
@app.route("/")
def home():
    return render_template("home.html")

# ---------------- CSV UPLOAD ----------------
@app.route("/upload", methods=["GET", "POST"])
def upload():

    global df

    if request.method == "POST":
        file = request.files["file"]

        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)
            df = pd.read_csv(filepath)

            return render_template("upload.html",
                                   message="File Uploaded Successfully!",
                                   rows=df.shape[0],
                                   cols=df.shape[1])

    return render_template("upload.html")

# ---------------- PREDICT ----------------
@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "POST":
        values = [
            float(request.form["Attendance"]),
            float(request.form["StudyHours"]),
            float(request.form["AssignmentScore"]),
            float(request.form["PreviousGPA"]),
            float(request.form["ParticipationLevel"]),
            float(request.form["InternetUsage"]),
            float(request.form["SleepHours"]),
            float(request.form["FamilySupportIndex"]),
            float(request.form["ExtraCurricular"])
        ]

        input_data = np.array([values])

        final_score = lr.predict(input_data)[0]
        pass_fail = dt.predict(input_data)[0]
        performance = knn.predict(input_data)[0]
        cluster = kmeans.predict(scaler.transform(input_data))[0]

        return render_template(
            "predict.html",
            score=round(final_score, 2),
            result=pass_fail,
            performance=performance,
            cluster=cluster
        )

    return render_template("predict.html")

# ---------------- VISUALIZE ----------------
@app.route("/visualize")
def visualize():

    performance_counts = (
        df["PerformanceCategory"]
        .value_counts()
        .reindex(["Low", "Medium", "High"])
        .fillna(0)
        .tolist()
    )

    # prepare cluster scatter data using first two numeric features
    # choose columns that were used to train kmeans (all predictors)
    features = df[["Attendance", "StudyHours"]].values
    # predict cluster labels for each point using same preprocessing
    scaled_full = scaler.transform(
        df.drop(["FinalExamScore", "PassFail", "PerformanceCategory"], axis=1)
    )
    labels = kmeans.predict(scaled_full)
    cluster_data = [
        {"x": float(pt[0]), "y": float(pt[1]), "cluster": int(lbl)}
        for pt, lbl in zip(features, labels)
    ]

    # centroids (use first two dimensions if kmeans was fit on scaled features)
    centers = kmeans.cluster_centers_
    centroid_points = []
    if centers is not None:
        for c in centers:
            # if scaler applied, reverse transform first two dims? we skip for simplicity
            centroid_points.append({"x": float(c[0]), "y": float(c[1])})

    return render_template(
        "visualize.html",
        performance_counts=performance_counts,
        cluster_data=cluster_data,
        centroid_points=centroid_points,
    )

# ---------------- METRICS ----------------
@app.route("/metrics")
def metrics():

    X = df.drop(["FinalExamScore", "PassFail", "PerformanceCategory"], axis=1)
    y_reg = df["FinalExamScore"]
    y_clf = df["PassFail"]
    y_knn = df["PerformanceCategory"]

    r2 = r2_score(y_reg, lr.predict(X))
    dt_acc = accuracy_score(y_clf, dt.predict(X))
    knn_acc = accuracy_score(y_knn, knn.predict(X))

    return render_template("metrics.html",
                           r2=round(r2, 4),
                           dt_acc=round(dt_acc, 4),
                           knn_acc=round(knn_acc, 4))

if __name__ == "__main__":
    app.run(debug=True)