import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
import scipy.fftpack as fft
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
cnn_model.fit(X_train_cnn, tf.keras.utils.to_categorical(y_train, num_classes=2), epochs=50, batch_size=32, verbose=1)

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

# DFT Test Implementation
def dft_test(bit_strings):
    n = len(bit_strings)
    fft_result = np.abs(fft.fft(bit_strings))[:n // 2]  # Use only half of the FFT output (real part)

    # Threshold for peaks
    threshold = np.sqrt(np.log(1 / 0.05) * n)

    # Compute expected number of peaks for random data
    expected_peaks = 0.95 * (n / 2)

    # Check if the sequence passes the DFT test
    passes_dft = np.sum(fft_result > threshold) <= expected_peaks
    return passes_dft, fft_result

# Run the DFT test on the dataset
bit_string_array = np.array(features_list).flatten()
passes_dft, fft_result = dft_test(bit_string_array)
print(f"DFT Test Passed: {passes_dft}")

# Plot the FFT result
plt.plot(fft_result)
plt.title('DFT Test - FFT Magnitude')
plt.axhline(y=np.sqrt(np.log(1 / 0.05) * len(bit_string_array)), color='r', linestyle='--')
plt.show()

# Linear Complexity Test Implementation
def berlekamp_massey_algorithm(bit_string):
    n = len(bit_string)
    c = [0] * n
    b = [0] * n
    c[0], b[0] = 1, 1
    l, m, i = 0, -1, 0

    for n in range(n):
        discrepancy = (bit_string[n] + sum([c[j] * bit_string[n - j - 1] for j in range(1, l + 1)])) % 2
        if discrepancy == 1:
            temp = c[:]
            for j in range(n - m):
                c[n - m + j] = (c[n - m + j] + b[j]) % 2
            if l <= n // 2:
                l = n + 1 - l
                m = n
                b = temp
    return l

def linear_complexity_test(bit_strings, threshold=0.01):
    complexities = [berlekamp_massey_algorithm(bit_string) for bit_string in bit_strings]
    mean_complexity = np.mean(complexities)
    passes_complexity_test = mean_complexity > len(bit_strings[0]) * threshold
    return passes_complexity_test, mean_complexity, complexities

# Run the linear complexity test
passes_lc, mean_complexity, complexities = linear_complexity_test(features_list)
print(f"Linear Complexity Test Passed: {passes_lc}")
print(f"Mean Linear Complexity: {mean_complexity}")
