import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the data
data_filePath = '1000qb_combined_output.txt'
data = pd.read_csv(data_filePath, sep=r'\s+', header=None)

# Display basic data information
print("Data shape:", data.shape)
print(data.head())

# Extract bit strings and labels
bit_strings = data.iloc[:, 0]  # Column with bit strings
labels = data.iloc[:, 1]  # Column with labels

# Map labels appropriately: PRNG -> 0, QRNG -> 1
label_mapping = {1: 0, 2: 1}  # Assuming 1 -> PRNG and 2 -> QRNG
labels = labels.map(label_mapping)
labels.fillna(labels.mode()[0], inplace=True)
labels = labels.astype(int)

# Convert bit strings to arrays of integers
features_list = []
for bit_string in bit_strings:
    bits = [int(bit) for bit in bit_string]
    features_list.append(bits)

# Convert to numpy array
features = np.array(features_list)
print("Features shape after parsing bit strings:", features.shape)

# Pre-processing: Normalize the data
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Shuffle and split the dataset
X, y = shuffle(features, labels, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# CNN Model Definition
def create_cnn_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.GlobalMaxPooling1D())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Reshape data for CNN input
X_train_cnn = X_train.reshape(-1, X_train.shape[1], 1)
X_test_cnn = X_test.reshape(-1, X_test.shape[1], 1)

# Build and Train CNN Model
cnn_model = create_cnn_model()
history = cnn_model.fit(X_train_cnn, tf.keras.utils.to_categorical(y_train, num_classes=2), epochs=50, batch_size=32, verbose=1, validation_data=(X_test_cnn, tf.keras.utils.to_categorical(y_test, num_classes=2)))

# Predict with CNN model
cnn_train_predictions = cnn_model.predict(X_train_cnn)
cnn_test_predictions = cnn_model.predict(X_test_cnn)

# Convert CNN predictions to labels (from probabilities)
y_train_cnn_pred = np.argmax(cnn_train_predictions, axis=1)
y_test_cnn_pred = np.argmax(cnn_test_predictions, axis=1)

# RandomForest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Predict using RandomForest
y_train_rf_pred = rf_model.predict(X_train)
y_test_rf_pred = rf_model.predict(X_test)

# Stacking: Use predictions from CNN and RandomForest as inputs to a Logistic Regression meta-model

# Stack the predictions for the training set
stacked_train_predictions = np.column_stack((y_train_cnn_pred, y_train_rf_pred))

# Stack the predictions for the test set
stacked_test_predictions = np.column_stack((y_test_cnn_pred, y_test_rf_pred))

# Train the Logistic Regression meta-model
meta_model = LogisticRegression(random_state=42)
meta_model.fit(stacked_train_predictions, y_train)

# Predict using the stacked model (meta-learner)
y_test_meta_pred = meta_model.predict(stacked_test_predictions)

# Evaluate the Stacking Ensemble Model
accuracy_stacked = accuracy_score(y_test, y_test_meta_pred)
precision_stacked = precision_score(y_test, y_test_meta_pred, zero_division=0)
recall_stacked = recall_score(y_test, y_test_meta_pred, zero_division=0)
f1_stacked = f1_score(y_test, y_test_meta_pred, zero_division=0)

print(f"\nStacking Ensemble Model Test Accuracy: {accuracy_stacked:.4f}")
print(f"Stacking Ensemble Model Precision: {precision_stacked:.4f}")
print(f"Stacking Ensemble Model Recall: {recall_stacked:.4f}")
print(f"Stacking Ensemble Model F1-Score: {f1_stacked:.4f}")

# Plot CNN training and validation accuracy over epochs
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('CNN Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
