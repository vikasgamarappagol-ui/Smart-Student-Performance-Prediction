import numpy as np
import pandas as pd

np.random.seed(42)
n = 1200

# Generate features
attendance = np.random.randint(60, 101, n)
study_hours = np.random.uniform(2, 10, n)
assignment_score = np.random.randint(50, 101, n)
previous_gpa = np.random.uniform(6, 10, n)
participation = np.random.randint(1, 6, n)
internet_usage = np.random.uniform(0, 6, n)
sleep_hours = np.random.uniform(5, 9, n)
family_support = np.random.randint(1, 6, n)
extra_curricular = np.random.randint(0, 2, n)

# 🔥 Very small noise for high R2
noise = np.random.normal(0, 1, n)

# Strong linear relationship
final_score = (
    0.4 * attendance +
    6 * study_hours +
    0.4 * assignment_score +
    4 * previous_gpa +
    2 * participation +
    2 * family_support -
    1.5 * internet_usage +
    1.5 * sleep_hours +
    3 * extra_curricular +
    noise
)

final_score = np.clip(final_score / 2, 0, 100)

# Classification Targets
pass_fail = np.where(final_score >= 40, "Pass", "Fail")

performance_category = pd.cut(
    final_score,
    bins=[0, 40, 70, 100],
    labels=["Low", "Medium", "High"]
)

# Create DataFrame
df = pd.DataFrame({
    "Attendance": attendance,
    "StudyHours": study_hours.round(2),
    "AssignmentScore": assignment_score,
    "PreviousGPA": previous_gpa.round(2),
    "ParticipationLevel": participation,
    "InternetUsage": internet_usage.round(2),
    "SleepHours": sleep_hours.round(2),
    "FamilySupportIndex": family_support,
    "ExtraCurricular": extra_curricular,
    "FinalExamScore": final_score.round(2),
    "PassFail": pass_fail,
    "PerformanceCategory": performance_category
})

df.to_csv("student_synthetic_data_final.csv", index=False)

print("Dataset Generated Successfully!")
print("Shape:", df.shape)
print(df.head())