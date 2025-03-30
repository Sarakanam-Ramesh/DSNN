import os
import numpy as np
import pandas as pd
import scipy.signal as signal
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import class_weight
import wfdb
from scipy.signal import find_peaks

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define paths
normal_dir = "D:/DSNN/data/edf"
abnormal_dir = "D:/DSNN/data/edf"

def load_record(record_path, record_name):
    """Load a record and its associated annotation."""
    record = wfdb.rdrecord(os.path.join(record_path, record_name))
    signals = record.p_signal.T
    return signals, record.fs

def extract_ecg_segments(signal, fs, segment_length_sec=10, overlap_ratio=0.5):
    """Extract segments from ECG signal with overlap."""
    segment_length = int(segment_length_sec * fs)
    step_size = int(segment_length * (1 - overlap_ratio))
    segments = []
    
    for i in range(0, len(signal) - segment_length + 1, step_size):
        segment = signal[i:i + segment_length]
        segments.append(segment)
    
    return segments

def calculate_heart_rate(ecg_signal, fs):
    """Calculate heart rate from ECG signal."""
    # Apply bandpass filter to isolate QRS complex frequencies
    lowcut = 5.0  # Hz
    highcut = 15.0  # Hz
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(2, [low, high], btype='band')
    filtered_ecg = signal.filtfilt(b, a, ecg_signal)
    
    # Find R-peaks
    r_peaks, _ = find_peaks(filtered_ecg, height=0.5*np.max(filtered_ecg), distance=int(0.5*fs))
    
    # Calculate heart rate
    if len(r_peaks) > 1:
        # Calculate RR intervals
        rr_intervals = np.diff(r_peaks) / fs  # in seconds
        
        # Convert to heart rate in BPM
        heart_rates = 60 / rr_intervals
        
        # Remove outliers (HR outside normal human range)
        valid_hr = heart_rates[(heart_rates >= 40) & (heart_rates <= 200)]
        
        if len(valid_hr) > 0:
            return np.mean(valid_hr)
    
    # Default return if calculation fails
    return 75.0  # Average adult resting heart rate

def extract_features(segments, fs):
    """Extract features from ECG segments."""
    features = []
    
    for segment in segments:
        # Basic statistics
        mean = np.mean(segment)
        std = np.std(segment)
        min_val = np.min(segment)
        max_val = np.max(segment)
        
        # Heart rate
        heart_rate = calculate_heart_rate(segment, fs)
        
        # Power spectral density features
        freqs, psd = signal.welch(segment, fs, nperseg=len(segment)//4)
        peak_freq = freqs[np.argmax(psd)]
        total_power = np.sum(psd)
        
        # Combine features
        feature_vector = np.array([mean, std, min_val, max_val, heart_rate, peak_freq, total_power])
        features.append(feature_vector)
    
    return np.array(features)

def load_data():
    """Load and prepare data for training."""
    # Load normal data
    normal_segments = []
    normal_fs_values = []
    
    for record_name in os.listdir(normal_dir):
        if not record_name.endswith('.dat'):
            continue
        record_name = record_name.replace('.dat', '')
        try:
            signals, fs = load_record(normal_dir, record_name)
            normal_fs_values.append(fs)
            
            for channel in signals:
                segments = extract_ecg_segments(channel, fs)
                normal_segments.extend(segments)
                
        except Exception as e:
            print(f"Error processing normal record {record_name}: {e}")
    
    # Load abnormal data
    abnormal_segments = []
    abnormal_fs_values = []
    
    for record_name in os.listdir(abnormal_dir):
        if not record_name.endswith('.dat'):
            continue
        record_name = record_name.replace('.dat', '')
        try:
            signals, fs = load_record(abnormal_dir, record_name)
            abnormal_fs_values.append(fs)
            
            for channel in signals:
                segments = extract_ecg_segments(channel, fs)
                abnormal_segments.extend(segments)
                
        except Exception as e:
            print(f"Error processing abnormal record {record_name}: {e}")
    
    # Perform data validation
    print(f"Normal data: {len(normal_segments)} segments, average fs: {np.mean(normal_fs_values):.2f} Hz")
    print(f"Abnormal data: {len(abnormal_segments)} segments, average fs: {np.mean(abnormal_fs_values):.2f} Hz")
    
    # Check for sampling rate consistency
    if max(normal_fs_values) != min(normal_fs_values) or max(abnormal_fs_values) != min(abnormal_fs_values):
        print("WARNING: Inconsistent sampling rates detected. Resampling may be required.")
    
    # Create labels
    normal_labels = np.zeros(len(normal_segments))
    abnormal_labels = np.ones(len(abnormal_segments))
    
    # Combine data
    X = np.vstack((normal_segments, abnormal_segments))
    y = np.hstack((normal_labels, abnormal_labels))
    
    # Check for data imbalance
    unique, counts = np.unique(y, return_counts=True)
    imbalance_ratio = max(counts) / min(counts)
    print(f"Class distribution: {dict(zip(unique, counts))}")
    print(f"Imbalance ratio: {imbalance_ratio:.2f}")
    
    # Check for outliers
    for i, segment in enumerate([normal_segments, abnormal_segments]):
        class_name = "Normal" if i == 0 else "Abnormal"
        segment_array = np.array(segment)
        mean_vals = np.mean(segment_array, axis=1)
        std_vals = np.std(segment_array, axis=1)
        
        print(f"{class_name} data statistics:")
        print(f"  Mean range: {np.min(mean_vals):.4f} to {np.max(mean_vals):.4f}")
        print(f"  Std range: {np.min(std_vals):.4f} to {np.max(std_vals):.4f}")
        
        # Detect potential outliers
        outlier_threshold = 3
        z_scores = (mean_vals - np.mean(mean_vals)) / np.std(mean_vals)
        outliers = np.abs(z_scores) > outlier_threshold
        if np.any(outliers):
            print(f"  WARNING: {np.sum(outliers)} potential outliers detected in {class_name} class.")
    
    return X, y

def build_model(input_shape):
    """Build the CNN model."""
    model = Sequential([
        # First Conv block
        Conv1D(64, 7, activation='relu', padding='same', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        # Second Conv block
        Conv1D(128, 5, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        # Third Conv block
        Conv1D(128, 3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # Fully connected layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    return model

# Load data
X, y = load_data()

# Split data with stratification to maintain class balance in all sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Print data split information
print(f"Training set: {X_train.shape}, {np.sum(y_train == 0)} normal, {np.sum(y_train == 1)} abnormal")
print(f"Validation set: {X_val.shape}, {np.sum(y_val == 0)} normal, {np.sum(y_val == 1)} abnormal")
print(f"Test set: {X_test.shape}, {np.sum(y_test == 0)} normal, {np.sum(y_test == 1)} abnormal")

# Calculate class weights
class_weights = class_weight.compute_class_weight('balanced',
                                                 classes=np.unique(y_train),
                                                 y=y_train)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
print(f"Class weights: {class_weight_dict}")

# Apply a maximum cap to avoid excessive weighting
max_weight = 10.0
for key in class_weight_dict:
    class_weight_dict[key] = min(class_weight_dict[key], max_weight)
print(f"Adjusted class weights: {class_weight_dict}")

# Build model
input_shape = (X_train.shape[1], 1)
model = build_model(input_shape)

# Reshape data for CNN
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Define learning rate schedule
initial_learning_rate = 1e-3
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.9,
    staircase=True)

# Compile model
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC()])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_auc', mode='max', save_best_only=True, verbose=1)

# Train model
batch_size = 32  # Reduced batch size
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=batch_size,
    callbacks=[early_stopping, model_checkpoint],
    class_weight=class_weight_dict,
    verbose=1
)

# Load best model
model.load_weights('best_model.h5')

# Evaluate model
def evaluate_model(model, X, y, set_name):
    y_pred_proba = model.predict(X)
    
    # Find optimal threshold using ROC curve
    fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"Optimal threshold for {set_name} set: {optimal_threshold:.4f}")
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    
    # Print metrics
    print(f"\nEvaluation on {set_name} set:")
    print(classification_report(y, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix ({set_name} set)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'confusion_matrix_{set_name}.png')
    plt.close()
    
    # Plot ROC curve
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve ({set_name} set)')
    plt.legend(loc="lower right")
    plt.savefig(f'roc_curve_{set_name}.png')
    plt.close()
    
    return y_pred, y_pred_proba, roc_auc

# Evaluate on validation set
val_pred, val_pred_proba, val_auc = evaluate_model(model, X_val, y_val, "validation")

# Evaluate on test set
test_pred, test_pred_proba, test_auc = evaluate_model(model, X_test, y_test, "test")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.close()

# Print model summary
model.summary()

# Perform k-fold cross validation for a more robust evaluation
def perform_kfold_cv(X, y, n_splits=5):
    from sklearn.model_selection import StratifiedKFold
    
    # Initialize variables to store results
    cv_scores = []
    cv_aucs = []
    fold = 1
    
    # Create KFold object
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for train_idx, test_idx in kfold.split(X, y):
        print(f"\nFold {fold}/{n_splits}")
        
        # Split data
        X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]
        
        # Further split training data to get validation set
        X_train_fold, X_val_fold, y_train_fold, y_val_fold = train_test_split(
            X_train_fold, y_train_fold, test_size=0.2, random_state=42, stratify=y_train_fold
        )
        
        # Reshape data
        X_train_fold = X_train_fold.reshape(X_train_fold.shape[0], X_train_fold.shape[1], 1)
        X_val_fold = X_val_fold.reshape(X_val_fold.shape[0], X_val_fold.shape[1], 1)
        X_test_fold = X_test_fold.reshape(X_test_fold.shape[0], X_test_fold.shape[1], 1)
        
        # Calculate class weights
        fold_class_weights = class_weight.compute_class_weight('balanced',
                                                             classes=np.unique(y_train_fold),
                                                             y=y_train_fold)
        fold_class_weight_dict = {i: min(weight, max_weight) for i, weight in enumerate(fold_class_weights)}
        
        # Build and compile model
        model_fold = build_model(input_shape)
        model_fold.compile(optimizer=optimizer,
                          loss='binary_crossentropy',
                          metrics=['accuracy', tf.keras.metrics.AUC()])
        
        # Train model
        history_fold = model_fold.fit(
            X_train_fold, y_train_fold,
            validation_data=(X_val_fold, y_val_fold),
            epochs=50,  # Reduced for cross-validation
            batch_size=batch_size,
            callbacks=[early_stopping],
            class_weight=fold_class_weight_dict,
            verbose=0
        )
        
        # Evaluate model
        fold_score = model_fold.evaluate(X_test_fold, y_test_fold, verbose=0)
        y_pred_proba_fold = model_fold.predict(X_test_fold)
        fpr, tpr, _ = roc_curve(y_test_fold, y_pred_proba_fold)
        fold_auc = auc(fpr, tpr)
        
        print(f"Fold {fold} - Loss: {fold_score[0]:.4f}, Accuracy: {fold_score[1]:.4f}, AUC: {fold_auc:.4f}")
        
        cv_scores.append(fold_score[1])
        cv_aucs.append(fold_auc)
        
        fold += 1
    
    # Print summary
    print("\nCross-Validation Results:")
    print(f"Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    print(f"AUC: {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}")

# Run cross-validation
print("\nPerforming k-fold cross-validation...")
perform_kfold_cv(X, y, n_splits=5)

# Feature importance analysis
def analyze_feature_importance():
    # Create a simpler version of the model for analysis
    analysis_model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    analysis_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    analysis_model.fit(X_train.reshape(X_train.shape[0], X_train.shape[1]), y_train, 
                      epochs=20, batch_size=batch_size, verbose=0)
    
    # Extract weights from the first layer
    weights = analysis_model.layers[0].get_weights()[0]
    
    # Calculate feature importance
    importance = np.abs(weights).mean(axis=1)
    
    # Create a time array for plotting
    time = np.arange(len(importance))
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    plt.plot(time, importance)
    plt.title('ECG Signal Feature Importance')
    plt.xlabel('Time Point')
    plt.ylabel('Importance')
    plt.grid(True)
    plt.savefig('feature_importance.png')
    plt.close()

# Run feature importance analysis
analyze_feature_importance()

print("Analysis completed.")
            
