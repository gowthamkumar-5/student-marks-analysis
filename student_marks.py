# 1. Import
import pandas as pd
import matplotlib.pyplot as plt

# 2. Load dataset
df = pd.read_csv("students.csv")

print("DATASET:")
print(df)

# 3. Data Cleaning
print("\nMissing Values:")
print(df.isnull().sum())

# 4. Feature Engineering
df["total"] = df["maths"] + df["science"] + df["english"]

# 5. Basic Analysis
print("\nAverage Total Marks:", df["total"].mean())

top_student = df.sort_values("total", ascending=False).head(1)
print("\nTop Student:")
print(top_student)

# Weak students
weak_students = df[df["total"] < 150]
print("\nWeak Students:")
print(weak_students[["name", "total"]])

print("\nTop 3 Students:")
print(df.sort_values("total", ascending=False).head(3))

# 6. Visualization
plt.figure(figsize=(10,5))
df.plot(x="name", y="total", kind="bar")
plt.title("Student Performance Analysis")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

df.plot(x="name", y="total", kind="bar")
import matplotlib.pyplot as plt
plt.title("Student Total Marks")
plt.show()

# 7. Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X = df[["maths", "science", "english", "attendance", "study_hours"]]
y = df["final_result"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("\nModel Accuracy:", accuracy)

# 8. Prediction
sample = [[80, 75, 85, 90, 3]]
prediction = model.predict(sample)

print("\nNew Student Prediction (1=Pass, 0=Fail):", prediction)

df.to_csv("final_output.csv", index=False)