import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# STEP 1: LOAD DATASET
# ============================================================

# Load Breast Cancer dataset
data = load_breast_cancer()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y = pd.Series(data.target)

print("Dataset shape:", X.shape)


# ============================================================
# STEP 2: TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)


# ============================================================
# STEP 3: BAGGING
# ============================================================

bag_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=50,
    random_state=42
)

# Train Bagging model
bag_model.fit(X_train, y_train)

# Make predictions
y_pred_bag = bag_model.predict(X_test)


# Display Bagging results
print("\n=== Bagging Results ===")

print("Accuracy:", accuracy_score(y_test, y_pred_bag))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_bag))


# ============================================================
# STEP 4: BOOSTING - ADABOOST
# ============================================================

boost_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    random_state=42
)

# Train AdaBoost model
boost_model.fit(X_train, y_train)

# Make predictions
y_pred_boost = boost_model.predict(X_test)


# Display Boosting results
print("\n=== Boosting Results (AdaBoost) ===")

print("Accuracy:", accuracy_score(y_test, y_pred_boost))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_boost))


# ============================================================
# STEP 5: CONFUSION MATRICES
# ============================================================

print(
    "\nConfusion Matrix - Bagging:\n",
    confusion_matrix(y_test, y_pred_bag)
)

print(
    "\nConfusion Matrix - Boosting:\n",
    confusion_matrix(y_test, y_pred_boost)
)