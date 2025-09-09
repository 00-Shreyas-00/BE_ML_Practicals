import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

# ------------------------------
# Step 1: Load Dataset
# ------------------------------
digits = load_digits()
X, y = digits.data, digits.target   # X: pixel features, y: digit labels (0–9)

print("Dataset shape:", X.shape)  # (1797, 64) → 8x8 pixel images

# ------------------------------
# Step 2: Preprocessing
# ------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # SVM works better with standardized data

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ------------------------------
# Step 3: Train SVM Classifier
# ------------------------------
# RBF kernel is default and works well for image data
svm_clf = SVC(kernel="rbf", C=10, gamma=0.001)
svm_clf.fit(X_train, y_train)

# ------------------------------
# Step 4: Predictions & Evaluation
# ------------------------------
y_pred = svm_clf.predict(X_test)

print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=digits.target_names,
            yticklabels=digits.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for SVM Digit Classifier")
plt.show()

# ------------------------------
# Step 5: Visualize Predictions
# ------------------------------
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(8, 8), cmap="gray")
    ax.set_title(f"Pred: {y_pred[i]}\nTrue: {y_test[i]}")
    ax.axis("off")
plt.show()
