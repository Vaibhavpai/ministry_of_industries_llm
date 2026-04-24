import kagglehub
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

# Download dataset
path = kagglehub.dataset_download("nalisha/job-salary-prediction-dataset")

# Load CSV
files = os.listdir(path)
csv_file = [f for f in files if f.endswith('.csv')][0]
df = pd.read_csv(os.path.join(path, csv_file))

# Preprocessing
df = df.dropna()

# Encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and target
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Convert salary to binary classes
threshold = y.mean()
y_class = np.where(y > threshold, 1, 0)

# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train SVM
model = SVC(kernel='rbf')
model.fit(X, y_class)

# ----------- TEST WITH ONE EXAMPLE -----------

# Take one sample from dataset (you can modify this)
test_sample = X[0].reshape(1, -1)

prediction = model.predict(test_sample)[0]

# Convert to label
label = "High Salary" if prediction == 1 else "Low Salary"

print("\nTest Example Prediction:")
print("Predicted Class:", label)

# Show actual for comparison
actual = "High Salary" if y_class[0] == 1 else "Low Salary"
print("Actual Class:", actual)