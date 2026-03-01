import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, accuracy_score

# Load dataset
df = pd.read_csv("student_synthetic_data_final.csv")

# Features
X = df.drop(["FinalExamScore", "PassFail", "PerformanceCategory"], axis=1)

# Targets
y_reg = df["FinalExamScore"]
y_clf = df["PassFail"]
y_knn = df["PerformanceCategory"]

# Train-test split
X_train, X_test, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

_, _, y_train_clf, y_test_clf = train_test_split(
    X, y_clf, test_size=0.2, random_state=42
)

_, _, y_train_knn, y_test_knn = train_test_split(
    X, y_knn, test_size=0.2, random_state=42
)

# -----------------------------
# 🔹 Linear Regression
# -----------------------------
lr = LinearRegression()
lr.fit(X_train, y_train_reg)

r2 = r2_score(y_test_reg, lr.predict(X_test))
print("Linear Regression R2 Score:", round(r2, 4))

# -----------------------------
# 🔹 Decision Tree (Pass/Fail)
# -----------------------------
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train_clf)

acc_dt = accuracy_score(y_test_clf, dt.predict(X_test))
print("Decision Tree Accuracy:", round(acc_dt, 4))

# -----------------------------
# 🔹 KNN (Performance Category)
# -----------------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train_knn)

acc_knn = accuracy_score(y_test_knn, knn.predict(X_test))
print("KNN Accuracy:", round(acc_knn, 4))

# -----------------------------
# 🔹 KMeans (Clustering)
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# -----------------------------
# Save Models
# -----------------------------
joblib.dump(lr, "linear_model.pkl")
joblib.dump(dt, "decision_tree.pkl")
joblib.dump(knn, "knn_model.pkl")
joblib.dump(kmeans, "kmeans_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n✅ All Models Trained & Saved Successfully!")