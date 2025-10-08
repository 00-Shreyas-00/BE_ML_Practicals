# Random Forest Classifier on Car Evaluation Dataset

# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 2: Load Dataset
# Make sure you download 'car.data' or 'car_evaluation.csv' from Kaggle and place it in the same folder
data = pd.read_csv("datasets/car_evaluation.csv")

print("First 5 rows of dataset:")
print(data.head())

data.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

# Step 3: Encode Categorical Features
encoder = LabelEncoder()
for col in data.columns:
    data[col] = encoder.fit_transform(data[col])

print("\nEncoded dataset:")
print(data.head())

# Step 4: Split dataset into features and target
X = data.drop("class", axis=1)  # features
y = data["class"]              # target variable

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 6: Predictions
y_pred = rf_model.predict(X_test)

# Step 7: Evaluate Model
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 8: Feature Importance (Optional)
import matplotlib.pyplot as plt

importances = rf_model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(8, 5))
plt.barh(feature_names, importances)
plt.title("Feature Importance in Random Forest")
plt.show()
