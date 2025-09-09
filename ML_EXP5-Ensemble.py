import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Ensemble Models
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# Base models for voting
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# Encode target labels as numbers
le = LabelEncoder()

# Load Iris dataset
df = pd.read_csv("datasets/Iris.csv")

# Drop unnecessary columns
df = df.drop(columns=["Id"])

# Features and target
X = df.drop(columns=["Species"])
y = df["Species"]
y = le.fit_transform(y)

# Split into train & test
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Define base learners
log_clf = LogisticRegression(random_state=42, max_iter=500)
knn_clf = KNeighborsClassifier(n_neighbors=5)
svm_clf = SVC(probability=True, random_state=42)

# Hard Voting
voting_hard = VotingClassifier(
    estimators=[('lr', log_clf), ('knn', knn_clf), ('svc', svm_clf)],
    voting='hard'
)

# Soft Voting
voting_soft = VotingClassifier(
    estimators=[('lr', log_clf), ('knn', knn_clf), ('svc', svm_clf)],
    voting='soft'
)

# Train
voting_hard.fit(X_train, y_train)
voting_soft.fit(X_train, y_train)

# Predictions
y_pred_hard = voting_hard.predict(X_test)
y_pred_soft = voting_soft.predict(X_test)


# AdaBoost
ada = AdaBoostClassifier(n_estimators=100, random_state=42)
ada.fit(X_train, y_train)
y_pred_ada = ada.predict(X_test)

# Gradient Boosting
gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gbm.fit(X_train, y_train)
y_pred_gbm = gbm.predict(X_test)

# XGBoost
xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)


def evaluate_model(name, y_true, y_pred):
    print(f"Model: {name}")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))
    print("-"*50)

# Evaluate all models
evaluate_model("Voting (Hard)", y_test, y_pred_hard)
evaluate_model("Voting (Soft)", y_test, y_pred_soft)
evaluate_model("AdaBoost", y_test, y_pred_ada)
evaluate_model("Gradient Boosting", y_test, y_pred_gbm)
evaluate_model("XGBoost", y_test, y_pred_xgb)
