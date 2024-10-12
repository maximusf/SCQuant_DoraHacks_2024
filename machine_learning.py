import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.combine import SMOTEENN
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

# Load the data
data_filePath = '1000qb_qwerty.txt'
data = pd.read_csv(data_filePath, sep=r'\s+', header=None)

# Display basic data information
print("Data shape:", data.shape)
print(data.head())

# Extract bit strings and labels
bit_strings = data.iloc[:, 0]  # Column with bit strings
labels = data.iloc[:, 1]  # Column with labels

# Map labels appropriately
label_mapping = {1: 0, 2: 1}  # Assuming 1 -> PRNG and 2 -> QRNG
labels = labels.map(label_mapping)
labels.fillna(labels.mode()[0], inplace=True)

# Convert bit strings to arrays of integers
features_list = []
for bit_string in bit_strings:
    bits = [int(bit) for bit in bit_string]
    features_list.append(bits)

# Convert to DataFrame
features = pd.DataFrame(features_list)
print("Features shape after parsing bit strings:", features.shape)


# Feature extraction function
def extract_features(row):
    measurements = row.values.astype(int)
    num_qubits = len(measurements)  # Should be 100
    features = {}

    # Add bit-wise features
    for qubit in range(num_qubits):
        bit_value = measurements[qubit]
        features[f'qubit_{qubit}_value'] = bit_value

    return pd.Series(features)


# Additional features (mean, variance, count of 1's and 0's)
def additional_features(bit_string):
    bit_array = np.array([int(bit) for bit in bit_string])
    return {
        'mean': np.mean(bit_array),
        'variance': np.var(bit_array),
        'ones_count': np.sum(bit_array),
        'zeros_count': len(bit_array) - np.sum(bit_array)
    }


# Apply feature extraction and additional features
features_extracted = features.apply(extract_features, axis=1)
additional_feature_df = bit_strings.apply(additional_features).apply(pd.Series)
features = pd.concat([features_extracted, additional_feature_df], axis=1)

print("Extracted features shape:", features.shape)

# Proceed with splitting the data
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, stratify=labels, random_state=42
)

# Apply SMOTE-ENN for better balancing
smote_enn = SMOTEENN(random_state=42)
X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train, y_train)

# Feature Selection using RFE
rfe_selector = RFE(estimator=RandomForestClassifier(random_state=42), n_features_to_select=50, step=10)
X_train_selected = rfe_selector.fit_transform(X_train_resampled, y_train_resampled)
X_test_selected = rfe_selector.transform(X_test)

# Convert labels to integers
y_train_resampled = y_train_resampled.astype(int)
y_test = y_test.astype(int)

# Train CatBoost model
catboost_model = CatBoostClassifier(iterations=100, learning_rate=0.05, depth=8, random_seed=42, verbose=0)
catboost_model.fit(X_train_selected, y_train_resampled)

# Predict on the test set
y_pred_catboost = catboost_model.predict(X_test_selected)

# Calculate evaluation metrics
accuracy_catboost = accuracy_score(y_test, y_pred_catboost)
precision_catboost = precision_score(y_test, y_pred_catboost, zero_division=0)
recall_catboost = recall_score(y_test, y_pred_catboost, zero_division=0)
f1_catboost = f1_score(y_test, y_pred_catboost, zero_division=0)

# Print CatBoost results
print(f"\nCatBoost Model Test Accuracy: {accuracy_catboost:.4f}")
print(f"CatBoost Model Precision: {precision_catboost:.4f}")
print(f"CatBoost Model Recall: {recall_catboost:.4f}")
print(f"CatBoost Model F1-Score: {f1_catboost:.4f}")

# Stacking Classifier: CatBoost, RandomForest, Logistic Regression
estimators = [
    ('catboost', CatBoostClassifier(iterations=100, learning_rate=0.05, depth=8, random_seed=42, verbose=0)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('lr', LogisticRegression(max_iter=1000))
]

stacking_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stacking_clf.fit(X_train_selected, y_train_resampled)

# Predict using Stacking Classifier
y_pred_stack = stacking_clf.predict(X_test_selected)

# Evaluate the Stacking Classifier
accuracy_stack = accuracy_score(y_test, y_pred_stack)
precision_stack = precision_score(y_test, y_pred_stack, zero_division=0)
recall_stack = recall_score(y_test, y_pred_stack, zero_division=0)
f1_stack = f1_score(y_test, y_pred_stack, zero_division=0)

# Print Stacking results
print(f"\nStacking Classifier Test Accuracy: {accuracy_stack:.4f}")
print(f"Stacking Classifier Precision: {precision_stack:.4f}")
print(f"Stacking Classifier Recall: {recall_stack:.4f}")
print(f"Stacking Classifier F1-Score: {f1_stack:.4f}")
