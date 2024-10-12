import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
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

# Hyperparameter tuning for CatBoost
# Define the CatBoost model
catboost_model = CatBoostClassifier(verbose=0, random_state=42)

# Define the parameter grid
param_dist = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'depth': [4, 6, 8, 10],
    'iterations': [100, 200, 300, 500],
    'l2_leaf_reg': [1, 3, 5, 7, 9],
    'bagging_temperature': [0.1, 0.2, 0.5, 1.0]
}

# Set up the RandomizedSearchCV with CatBoost
random_search = RandomizedSearchCV(
    catboost_model,
    param_distributions=param_dist,
    n_iter=20,  # Number of random combinations to try
    scoring='accuracy',
    cv=3,  # 3-fold cross-validation
    random_state=42,
    n_jobs=-1,  # Use all available cores
    verbose=1
)

# Fit the random search model
random_search.fit(X_train_selected, y_train_resampled)

# Print the best parameters and the best score
print(f"Best parameters: {random_search.best_params_}")
print(f"Best cross-validation accuracy: {random_search.best_score_:.4f}")

# Predict using the best model found by RandomizedSearchCV
best_catboost_model = random_search.best_estimator_

y_pred_best_catboost = best_catboost_model.predict(X_test_selected)

# Calculate evaluation metrics
accuracy_best = accuracy_score(y_test, y_pred_best_catboost)
precision_best = precision_score(y_test, y_pred_best_catboost, zero_division=0)
recall_best = recall_score(y_test, y_pred_best_catboost, zero_division=0)
f1_best = f1_score(y_test, y_pred_best_catboost, zero_division=0)

# Print results
print(f"\nTuned CatBoost Model Test Accuracy: {accuracy_best:.4f}")
print(f"Tuned CatBoost Model Precision: {precision_best:.4f}")
print(f"Tuned CatBoost Model Recall: {recall_best:.4f}")
print(f"Tuned CatBoost Model F1-Score: {f1_best:.4f}")

# Stacking Classifier: CatBoost, RandomForest, Logistic Regression
estimators = [
    ('catboost', best_catboost_model),
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
