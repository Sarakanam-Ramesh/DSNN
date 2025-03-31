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

def process_single_file(base_path, file_name, using_sliding_window=False):
    edf_path = os.path.join(base_path, file_name + ".edf")
    qrs_path = os.path.join(base_path, file_name)  # QRS file path without extension
    
    print(f"\nProcessing file: {file_name}")
    
    f = None
    try:
        # Load EDF file
        try:
            f = pyedflib.EdfReader(edf_path)
        except Exception as e:
            print(f"Error loading file {file_name}: {e}")
            return None
        
        # Check channels
        n_channels = f.signals_in_file
        signal_labels = f.getSignalLabels()
        print(f"Number of channels in the file: {n_channels}")
        print("Channel labels:", signal_labels)
        
        # Determine lead configuration
        lead_config = determine_lead_configuration(signal_labels)
        print(f"Detected lead configuration: {lead_config}")
        
        # Read the first two channels as leads for DSNN
        if n_channels < 2:
            print(f"File {file_name} does not have at least 2 channels, skipping")
            return None
            
        lead1 = f.readSignal(0)  # First channel
        lead2 = f.readSignal(1)  # Second channel
        
        # Validate lead lengths
        if len(lead1) != len(lead2):
            print(f"Warning: Lead lengths differ! Lead1: {len(lead1)}, Lead2: {len(lead2)}")
            # Use the shorter length to avoid index errors
            min_length = min(len(lead1), len(lead2))
            lead1 = lead1[:min_length]
            lead2 = lead2[:min_length]
            
        print(f"Lead 1 ({signal_labels[0]}) length: {len(lead1)}")
        print(f"Lead 2 ({signal_labels[1]}) length: {len(lead2)}")
        
        # Get sampling frequency
        fs = f.getSampleFrequency(0)
        print(f"Sampling frequency: {fs} Hz")
        
        # Try to load QRS annotations if available
        using_qrs = False
        r_peaks = []
        if not using_sliding_window:
            try:
                print("Attempting to load QRS annotations...")
                ann = wfdb.rdann(qrs_path, 'qrs')
                r_peaks = ann.sample  # R-peak sample locations
                print(f"Found {len(r_peaks)} R-peaks in the QRS file")
                using_qrs = True
            except Exception as e:
                print(f"Could not load QRS annotations: {e}")
                print("Will proceed without QRS annotations")
        
        # Extract segments based on whether QRS annotations are available
        segments = []
        try:
            if using_qrs and len(r_peaks) > 0:
                print("Extracting segments centered around R-peaks...")
                segments = extract_segments_around_rpeaks(lead1, lead2, r_peaks)
            else:
                print("Extracting segments using sliding window...")
                segments = extract_segments_sliding_window(lead1, lead2)
            
            print(f"Extracted {len(segments)} segments of length 24 from the ECG data")
        except Exception as e:
            print(f"Error during segment extraction: {e}")
            segments = []
        
        # Calculate heart rate if QRS annotations are available
        heart_rate = None
        hr_categories = []
        
        if using_qrs and len(r_peaks) > 1:
            try:
                heart_rate = calculate_heart_rate(r_peaks, fs)
                if heart_rate is not None:
                    print(f"\nCalculated heart rate: {heart_rate:.1f} BPM")
                    hr_categories = classify_heart_rate(heart_rate)
                    print("Possible categories based on heart rate:")
                    for category in hr_categories:
                        print(f"- {category}")
            except Exception as e:
                print(f"Error calculating heart rate: {e}")
        else:
            print("\nHeart rate calculation requires QRS annotations, which are not available")
        
        file_info = {
            "file_name": file_name,
            "segments": segments,
            "heart_rate": heart_rate,
            "hr_categories": hr_categories,
            "signal_labels": signal_labels,
            "lead_config": lead_config,
            "fs": fs,
            "using_qrs": using_qrs,
            "r_peaks": r_peaks,
            # Store signal info but not the full signals to save memory
             "lead1": lead1,  # commented out to save memory
             "lead2": lead2   # commented out to save memory
        }
        
        return file_info
        
    except Exception as e:
        print(f"Unexpected error processing file {file_name}: {e}")
        return None
    finally:
        # Ensure file is closed even if an exception occurs
        if f is not None:
            f.close()

def determine_lead_configuration(signal_labels):
    """
    Determine the lead configuration based on channel labels
    Returns a dictionary with lead configuration and placement information
    """
    # Handle empty input
    if not signal_labels:
        return {
            "type": "Unknown configuration",
            "description": "No signal labels provided",
            "lead_placement": {}
        }
        
    # Convert labels to lowercase for case-insensitive matching
    # Handle potential non-string labels and clean up whitespace
    lower_labels = []
    for label in signal_labels:
        if isinstance(label, str):
            lower_labels.append(label.lower().strip())
        else:
            lower_labels.append("")
    
    # Standard 12-lead ECG configuration detection with exact matching
    # Use more specific matching to avoid false positives
    standard_leads = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
    
    # More robust lead detection by checking for exact matches or common prefixes/suffixes
    found_leads = []
    for std_lead in standard_leads:
        for label in lower_labels:
            # Match exact labels or labels with common separators like "lead i", "i-", "lead-i"
            if (label == std_lead or 
                f"lead {std_lead}" in label or 
                f"lead-{std_lead}" in label or
                f"{std_lead}-" in label or 
                f"-{std_lead}" in label or
                label.endswith(f" {std_lead}")):
                found_leads.append(std_lead)
                break
    
    # Define configurations based on detected leads
    if len(found_leads) >= 10:  # If most standard leads are found
        config = {
            "type": "Standard 12-lead ECG",
            "description": "Standard clinical 12-lead configuration",
            "lead_placement": {
                "Limb leads": "I, II, III (frontal plane)",
                "Augmented limb leads": "aVR, aVL, aVF (frontal plane)",
                "Precordial leads": "V1-V6 (horizontal plane across chest)"
            }
        }
    elif 'i' in found_leads and 'ii' in found_leads:
        config = {
            "type": "3-lead ECG",
            "description": "Basic cardiac monitoring with leads I and II",
            "lead_placement": {
                "Lead I": "Right arm to left arm (lateral)",
                "Lead II": "Right arm to left leg (inferior)"
            }
        }
    elif any('ml' in label for label in lower_labels) or any('mr' in label for label in lower_labels):
        config = {
            "type": "Modified chest lead system",
            "description": "Monitoring leads optimized for ambulatory recording",
            "lead_placement": {
                "ML": "Modified chest leads for continuous monitoring",
                "MR": "Modified chest leads for continuous monitoring"
            }
        }
    else:
        # If no standard configuration detected, create a generic description
        # Make sure we handle the case where there might be fewer than 2 labels
        primary_label = signal_labels[0] if len(signal_labels) > 0 else "Unknown"
        secondary_label = signal_labels[1] if len(signal_labels) > 1 else "Unknown"
        
        config = {
            "type": "Custom ECG configuration",
            "description": f"Non-standard lead configuration: {', '.join(signal_labels[:5])}",
            "lead_placement": {
                primary_label: "Primary recording lead",
                secondary_label: "Secondary recording lead"
            }
        }
    
    return config

def extract_segments_around_rpeaks(lead1, lead2, r_peaks, segment_length=24, offset=None):
    """Extract segments centered around R-peaks
    
    Parameters:
    -----------
    lead1, lead2 : array-like
        The ECG lead signals
    r_peaks : array-like
        Indices of R-peaks
    segment_length : int, default=24
        Length of segments to extract
    offset : int, default=None
        Offset from R-peak for the start of segment. If None, segments will be centered on R-peaks.
    
    Returns:
    --------
    np.ndarray
        Array of extracted segments with shape (num_segments, 2, segment_length)
    """
    segments = []
    
    # If offset is None, center the segments on R-peaks
    if offset is None:
        offset = segment_length // 2
    
    for peak in r_peaks:
        # Calculate segment boundaries
        start = peak - offset
        end = start + segment_length
        
        # Skip if segment would go out of bounds
        if start < 0 or end > len(lead1):
            continue
            
        # Verify both leads have sufficient data
        if end > len(lead2):
            continue
            
        # Extract segment from both leads
        segment1 = lead1[start:end]
        segment2 = lead2[start:end]
        
        # Check that segments have the expected length
        if len(segment1) != segment_length or len(segment2) != segment_length:
            continue
        
        # Stack the two leads
        segment = np.stack([segment1, segment2])
        segments.append(segment)
    
    return np.array(segments)
def extract_segments_sliding_window(lead1, lead2, segment_length=24, stride=12):
    """Extract segments using sliding window
    
    Parameters:
    -----------
    lead1, lead2 : array-like
        The ECG lead signals
    segment_length : int, default=24
        Length of segments to extract
    stride : int, default=12
        Step size between consecutive segments
    
    Returns:
    --------
    np.ndarray
        Array of extracted segments with shape (num_segments, 2, segment_length)
    """
    segments = []
    
    # Convert inputs to numpy arrays if they aren't already
    lead1 = np.asarray(lead1)
    lead2 = np.asarray(lead2)
    
    # Make sure both leads have the same length
    min_length = min(len(lead1), len(lead2))
    lead1 = lead1[:min_length]
    lead2 = lead2[:min_length]
    
    # Extract segments with a specified stride
    for i in range(0, min_length - segment_length + 1, stride):
        segment1 = lead1[i:i+segment_length]
        segment2 = lead2[i:i+segment_length]
        
        # Verify segments have the expected length
        if len(segment1) != segment_length or len(segment2) != segment_length:
            continue
            
        # Stack the two leads
        segment = np.stack([segment1, segment2])
        segments.append(segment)
    
    # Handle the case where no segments were extracted
    if not segments:
        return np.empty((0, 2, segment_length))
    
    return np.array(segments)

def calculate_heart_rate(r_peaks, fs):
    """Calculate heart rate in BPM from R-peak locations
    
    Parameters:
    -----------
    r_peaks : array-like
        Indices of R-peaks
    fs : float
        Sampling frequency in Hz
    
    Returns:
    --------
    float or None
        Median heart rate in BPM, or None if fewer than 2 R-peaks
    """
    if len(r_peaks) < 2:
        return None
    
    # Ensure r_peaks is sorted
    r_peaks = np.sort(np.array(r_peaks))
    
    # Calculate RR intervals in samples
    rr_intervals = np.diff(r_peaks)
    
    # Remove physiologically impossible intervals (could be detection errors)
    # Normal heart rates are between 20-220 BPM
    min_rr_samples = fs * 60 / 220  # Fastest acceptable heart rate (220 BPM)
    max_rr_samples = fs * 60 / 20   # Slowest acceptable heart rate (20 BPM)
    
    valid_intervals = (rr_intervals >= min_rr_samples) & (rr_intervals <= max_rr_samples)
    filtered_intervals = rr_intervals[valid_intervals]
    
    # If no valid intervals remain, return None
    if len(filtered_intervals) == 0:
        return None
    
    # Convert to seconds
    rr_seconds = filtered_intervals / fs
    
    # Calculate instantaneous heart rates
    inst_hr = 60 / rr_seconds
    
    # Return median heart rate (more robust than mean)
    return float(np.median(inst_hr))

def classify_heart_rate(bpm):
    """Classify heart rate based on the provided categories
    
    Parameters:
    -----------
    bpm : float or None
        Heart rate in beats per minute
    
    Returns:
    --------
    list
        List of matching categories, or empty list if bpm is None
    """
    # Handle None or invalid input
    if bpm is None or not np.isfinite(bpm):
        return []
    
    categories = {
        "Over Exercised person": (150, 190),
        "Fully Anxiety person": (100, 160),
        "Fully Depressed person": (50, 70),
        "Normal Healthy Person": (60, 100),
        "High BP person": (80, 120),
        "Low BP person": (50, 75),
        "Stressed person": (80, 130),
        "Fevered or illness person": (80, 120),
        "Stimulant (drugs) person": (90, 160),
        "Dehydrated person": (100, 140)
    }
    
    # Find all matching categories
    matches = []
    for category, (min_bpm, max_bpm) in categories.items():
        if min_bpm <= bpm <= max_bpm:
            matches.append(category)
    
    return matches

def is_normal_ecg(predictions):
    """
    Determine if an ECG is normal based on DSNN predictions
    
    Parameters:
    -----------
    predictions : array-like
        Array of class predictions for each segment
        
    Returns:
    --------
    tuple
        (is_normal, confidence) where:
        - is_normal: boolean indicating if ECG is classified as normal
        - confidence: percentage value representing confidence in the classification
    """
    # Ensure predictions is a numpy array
    predictions = np.asarray(predictions)
    
    if len(predictions) == 0:
        return False, 0.0
    
    # Class 0 represents Normal Sinus Rhythm
    normal_class = 0
    
    # Count occurrences of normal rhythm
    normal_count = np.sum(predictions == normal_class)
    total_segments = len(predictions)
    
    # Calculate percentage of normal segments
    normal_percentage = (normal_count / total_segments) * 100
    
    # If more than 80% of segments are classified as normal, consider the ECG normal
    is_normal = normal_percentage >= 80
    
    # Calculate confidence based on majority class
    confidence = normal_percentage if is_normal else (100 - normal_percentage)
    
    # Ensure confidence is a float
    return bool(is_normal), float(confidence)

def classify_abnormality_type(predictions, class_definitions):
    """
    Classify the type of abnormality based on DSNN predictions
    
    Parameters:
    -----------
    predictions : array-like
        Array of class predictions for each segment
    class_definitions : dict
        Dictionary mapping class IDs to class names
        
    Returns:
    --------
    dict
        Dictionary with abnormality types and their prevalence
        Each key is a class name, and the value is a dict with count and percentage
    """
    # Ensure predictions is a numpy array
    predictions = np.asarray(predictions)
    
    if len(predictions) == 0:
        return {}
    
    # Remove normal class from analysis
    abnormal_predictions = predictions[predictions != 0]
    
    if len(abnormal_predictions) == 0:
        return {}
    
    # Count occurrences of each abnormal class
    abnormality_types = {}
    
    # Handle potential class_definitions issues
    if class_definitions is None or not isinstance(class_definitions, dict):
        class_definitions = {}
    
    # Get counts of unique classes
    unique_classes, counts = np.unique(abnormal_predictions, return_counts=True)
    
    for cls, count in zip(unique_classes, counts):
        percentage = (count / len(abnormal_predictions)) * 100
        # Convert class number to int to ensure consistent dictionary lookup
        class_id = int(cls)
        class_name = class_definitions.get(class_id, f"Unknown Class {class_id}")
        abnormality_types[class_name] = {
            "count": int(count),
            "percentage": float(percentage)
        }
    
    return abnormality_types

def visualize_normal_abnormal_segments(segments, predictions, num_samples=3, random_seed=42):
    """
    Visualize example normal and abnormal ECG segments
    
    Parameters:
    -----------
    segments : array-like
        Array of ECG segments with shape (n_segments, 2, segment_length)
    predictions : array-like
        Array of class predictions for each segment
    num_samples : int, default=3
        Number of normal and abnormal samples to display
    random_seed : int or None, default=42
        Seed for random number generator, set to None for truly random selection
        
    Returns:
    --------
    matplotlib.figure.Figure or None
        Figure object containing the plots, or None if visualization couldn't be created
    """
    # Ensure inputs are numpy arrays
    segments = np.asarray(segments)
    predictions = np.asarray(predictions)
    
    # Check for empty inputs
    if len(segments) == 0 or len(predictions) == 0:
        print("Could not visualize segments: empty input arrays")
        return None
        
    # Check if segments and predictions match in length
    if len(segments) != len(predictions):
        print(f"Could not visualize segments: mismatched lengths (segments: {len(segments)}, predictions: {len(predictions)})")
        return None
    
    # Get indices of normal and abnormal segments
    normal_indices = np.where(predictions == 0)[0]
    abnormal_indices = np.where(predictions != 0)[0]
    
    # If we don't have both normal and abnormal segments, return
    if len(normal_indices) == 0 or len(abnormal_indices) == 0:
        print("Could not visualize normal/abnormal segments: missing either normal or abnormal examples")
        return None
    
    # Set random seed for reproducibility if specified
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Select random samples
    if len(normal_indices) >= num_samples:
        normal_samples = np.random.choice(normal_indices, num_samples, replace=False)
    else:
        normal_samples = normal_indices
        
    if len(abnormal_indices) >= num_samples:
        abnormal_samples = np.random.choice(abnormal_indices, num_samples, replace=False)
    else:
        abnormal_samples = abnormal_indices
    
    # Create visualization
    fig, axs = plt.subplots(num_samples, 4, figsize=(16, 3*num_samples))
    
    # Handle case where num_samples = 1
    if num_samples == 1:
        axs = axs.reshape(1, 4)
    
    # Plot normal segments
    for i, idx in enumerate(normal_samples):
        if i < num_samples:
            # Plot lead 1
            axs[i, 0].plot(segments[idx][0], 'g-')
            axs[i, 0].set_title(f"Normal - Lead 1 (Segment {idx})")
            axs[i, 0].set_ylabel("Amplitude")
            
            # Plot lead 2
            axs[i, 1].plot(segments[idx][1], 'g-')
            axs[i, 1].set_title(f"Normal - Lead 2 (Segment {idx})")
    
    # Plot abnormal segments
    for i, idx in enumerate(abnormal_samples):
        if i < num_samples:
            # Plot lead 1
            axs[i, 2].plot(segments[idx][0], 'r-')
            axs[i, 2].set_title(f"Abnormal - Lead 1 (Segment {idx}, Class {predictions[idx]})")
            
            # Plot lead 2
            axs[i, 3].plot(segments[idx][1], 'r-')
            axs[i, 3].set_title(f"Abnormal - Lead 2 (Segment {idx}, Class {predictions[idx]})")
    
    plt.tight_layout()
    return fig

def generate_normality_report(file_info, class_definitions):
    """
    Generate a detailed report on ECG normality/abnormality
    
    Parameters:
    -----------
    file_info : dict
        Dictionary containing ECG file information including predictions and metadata
    class_definitions : dict
        Dictionary mapping class IDs to class names
        
    Returns:
    --------
    dict
        Detailed report on ECG normality/abnormality
    """
    # Check for required keys
    if 'predictions' not in file_info:
        return {
            "error": "Missing predictions in file_info",
            "file_name": file_info.get('file_name', "Unknown")
        }
        
    # Get file name with error handling
    file_name = file_info.get('file_name', "Unknown")
    
    # Get predictions and ensure it's a numpy array
    predictions = np.asarray(file_info['predictions'])
    
    # Handle empty predictions
    if len(predictions) == 0:
        return {
            "file_name": file_name,
            "error": "No predictions available",
            "is_normal": False,
            "confidence": 0.0
        }
    
    # Get normality assessment
    is_normal, confidence = is_normal_ecg(predictions)
    
    # Calculate counts and percentages
    normal_count = int(np.sum(predictions == 0))
    abnormal_count = int(np.sum(predictions != 0))
    total_segments = len(predictions)
    
    # Calculate percentages safely
    if total_segments > 0:
        normal_percentage = float((normal_count / total_segments) * 100)
        abnormal_percentage = float((abnormal_count / total_segments) * 100)
    else:
        normal_percentage = 0.0
        abnormal_percentage = 0.0
    
    # Prepare the report
    report = {
        "file_name": file_name,
        "is_normal": bool(is_normal),
        "confidence": float(confidence),
        "normal_segments_count": normal_count,
        "normal_segments_percentage": normal_percentage,
        "abnormal_segments_count": abnormal_count,
        "abnormal_segments_percentage": abnormal_percentage,
        "abnormality_types": classify_abnormality_type(predictions, class_definitions),
        "heart_rate": file_info.get('heart_rate', None),
        "hr_categories": file_info.get('hr_categories', [])
    }
    
    # Add risk assessment based on abnormality types and heart rate
    risk_level = "Low"
    risk_factors = []
    
    # Assess risk based on abnormality prevalence
    if abnormal_percentage > 50:
        risk_level = "High"
        risk_factors.append(f"Majority of ECG segments ({abnormal_percentage:.1f}%) show abnormalities")
    elif abnormal_percentage > 20:
        risk_level = "Moderate"
        risk_factors.append(f"Significant proportion of ECG segments ({abnormal_percentage:.1f}%) show abnormalities")
    
    # Assess risk based on heart rate - with type checking
    hr = file_info.get('heart_rate')
    if hr is not None and isinstance(hr, (int, float)) and np.isfinite(hr):
        if hr > 100:
            risk_factors.append(f"Elevated heart rate ({float(hr):.1f} BPM)")
            if risk_level == "Low":
                risk_level = "Moderate"
        elif hr < 60:
            risk_factors.append(f"Low heart rate ({float(hr):.1f} BPM)")
            if risk_level == "Low":
                risk_level = "Moderate"
    
    # Assess risk based on abnormality types
    high_risk_conditions = ["Ventricular Arrhythmia", "ST Segment Abnormality"]
    for condition in high_risk_conditions:
        abnormality_info = report["abnormality_types"].get(condition, {})
        percentage = abnormality_info.get("percentage", 0)
        if percentage > 10:
            risk_factors.append(f"Presence of {condition} ({float(percentage):.1f}%)")
            risk_level = "High"
    
    report["risk_level"] = risk_level
    report["risk_factors"] = risk_factors
    
    return report

# Add imports for random seed control
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import pyedflib
from dsnn_example import DSNN, DSNNSystem

# Set fixed seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------- MAIN EXECUTION CODE -------

# Set base path for all files
base_path = "d:/DSNN/data/edf/"

# List of file names to process (without extension)
file_names = ["1", "2", "3", "4", "5"]  # Replace with your actual file names

# Force using sliding window for all files? Set to True if you don't have QRS annotations
use_sliding_window = False

# Process all files
print("Processing multiple ECG datasets...")
all_file_info = []
for file_name in file_names:
    file_info = process_single_file(base_path, file_name, use_sliding_window)
    if file_info is not None:
        all_file_info.append(file_info)

print(f"\nSuccessfully processed {len(all_file_info)} out of {len(file_names)} files")

# ------- PART 2: RUNNING DSNN ALGORITHM ON MULTIPLE DATASETS -------

# Initialize the DSNN model
print("\nInitializing DSNN model...")
# Force consistent device usage instead of dynamic selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize model with parameters that match your data
model = DSNN(input_channels=2, sequence_length=24, num_classes=6)
model.to(device)
dsnn_system = DSNNSystem(model)

# Define class definitions for consistent reference
class_definitions = {
    0: "Normal Sinus Rhythm",
    1: "Atrial Fibrillation",
    2: "Ventricular Arrhythmia",
    3: "Conduction Block",
    4: "Premature Contraction",
    5: "ST Segment Abnormality"
}

# Set model to evaluation mode
model.eval()

# Process each dataset
normality_reports = []
for idx, file_info in enumerate(all_file_info):
    print(f"\n{'='*60}")
    print(f"PROCESSING DATASET {idx+1}: {file_info['file_name']}")
    print(f"{'='*60}")
    print(f"Lead Configuration: {file_info['lead_config']['type']}")
    print(f"Lead Placement: {file_info['lead_config']['description']}")
    
    # Convert to PyTorch tensors
    segments = file_info['segments']
    X = torch.FloatTensor(segments).unsqueeze(2)  # Add channel dimension
    print(f"Input tensor shape: {X.shape}")
    
    # Visualize a few samples before testing
    print("Visualizing sample segments...")
    plt.figure(figsize=(12, 6))
    for i in range(min(3, len(segments))):
        plt.subplot(3, 2, i*2+1)
        plt.plot(segments[i][0])
        plt.title(f"Sample {i+1}, {file_info['signal_labels'][0]}")
        plt.xlabel("Sample Points (time)")
        plt.ylabel("Amplitude (mV)")
        
        plt.subplot(3, 2, i*2+2)
        plt.plot(segments[i][1])
        plt.title(f"Sample {i+1}, {file_info['signal_labels'][1]}")
        plt.xlabel("Sample Points (time)")
        plt.ylabel("Amplitude (mV)")
    
    plt.tight_layout()
    plt.suptitle(f"ECG Segments from {file_info['file_name']} - {file_info['lead_config']['type']}", fontsize=16)
    plt.subplots_adjust(top=0.9)
    
    # Save plot instead of showing it during processing
    plt.savefig(f"{file_info['file_name']}_sample_segments.png")
    plt.close()
    
    # Process segments in batches with more deterministic approach
    print("\nRunning DSNN algorithm on the data...")
    batch_size = 32
    predictions = []
    
    with torch.no_grad():
        # Ensure fixed order of batch processing
        indices = list(range(0, len(X), batch_size))
        for i in indices:
            batch = X[i:i+batch_size].to(device)
            batch_preds = dsnn_system.process_ecg(batch)
            predictions.extend(batch_preds.cpu().numpy())
    
    predictions = np.array(predictions)
    print(f"Generated {len(predictions)} predictions")
    
    # Store predictions in file_info
    file_info['predictions'] = predictions
    
    # Analyze the predictions
    print("\nAnalyzing predictions:")
    unique_classes, counts = np.unique(predictions, return_counts=True)
    file_info['unique_classes'] = unique_classes
    file_info['counts'] = counts
    
    for cls, count in zip(unique_classes, counts):
        percentage = (count / len(predictions)) * 100
        class_name = class_definitions.get(cls, f"Class {cls}")
        print(f"{class_name}: {count} segments ({percentage:.2f}%)")
    
    # NEW: Generate normality report
    normality_report = generate_normality_report(file_info, class_definitions)
    normality_reports.append(normality_report)
    
    # Print normality report
    print("\nECG Normality Assessment:")
    print(f"Overall Classification: {'NORMAL' if normality_report['is_normal'] else 'ABNORMAL'} " +
          f"(Confidence: {normality_report['confidence']:.1f}%)")
    print(f"Normal Segments: {normality_report['normal_segments_count']} " +
          f"({normality_report['normal_segments_percentage']:.1f}%)")
    print(f"Abnormal Segments: {normality_report['abnormal_segments_count']} " +
          f"({normality_report['abnormal_segments_percentage']:.1f}%)")
    
    if normality_report['abnormality_types']:
        print("\nAbnormality Types Detected:")
        for abnormality, stats in normality_report['abnormality_types'].items():
            print(f"- {abnormality}: {stats['count']} segments ({stats['percentage']:.1f}%)")
    
    print(f"\nRisk Level: {normality_report['risk_level']}")
    if normality_report['risk_factors']:
        print("Risk Factors:")
        for factor in normality_report['risk_factors']:
            print(f"- {factor}")
    
    # NEW: Visualize normal vs abnormal segments
    if normality_report['normal_segments_count'] > 0 and normality_report['abnormal_segments_count'] > 0:
        print("\nVisualizing normal vs abnormal ECG segments...")
        fig = visualize_normal_abnormal_segments(segments, predictions)
        # Save figure instead of showing it
        plt.savefig(f"{file_info['file_name']}_normal_vs_abnormal.png")
        plt.close(fig)
    
    # Visualize class distribution with normal/abnormal highlighting
    plt.figure(figsize=(12, 6))
    classes = [class_definitions.get(c, f"Class {c}") for c in unique_classes]
    colors = ['green' if c == 0 else 'red' for c in unique_classes]
    bars = plt.bar(classes, counts, color=colors)
    
    # Add count labels on top of bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{count}', ha='center', va='bottom')
    
    plt.xlabel('ECG Classification Classes')
    plt.ylabel('Number of Segments')
    plt.title(f'Distribution of Normal vs Abnormal Patterns - {file_info["file_name"]}')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Add legend
    normal_patch = plt.Rectangle((0, 0), 1, 1, fc="green", edgecolor='none')
    abnormal_patch = plt.Rectangle((0, 0), 1, 1, fc="red", edgecolor='none')
    plt.legend([normal_patch, abnormal_patch], ["Normal", "Abnormal"], loc="upper right")
    
    # Save figure instead of showing it
    plt.savefig(f"{file_info['file_name']}_class_distribution.png")
    plt.close()
    
    # Create a heart rate category visualization if heart rate is available
    if file_info['heart_rate'] is not None:
        heart_rate = file_info['heart_rate']
        hr_categories = file_info['hr_categories']
        
        categories_data = {
            "Over Exercised person": (150, 190),
            "Fully Anxiety person": (100, 160),
            "Fully Depressed person": (50, 70),
            "Normal Healthy Person": (60, 100),
            "High BP person": (80, 120),
            "Low BP person": (50, 75),
            "Stressed person": (80, 130),
            "Fevered or illness person": (80, 120),
            "Stimulant (drugs) person": (90, 160),
            "Dehydrated person": (100, 140)
        }
        
        # Create range visualization with heart rate marker
        plt.figure(figsize=(12, 8))
        category_names = list(categories_data.keys())
        y_positions = range(len(category_names))
        
        # Plot ranges as horizontal bars with normal/abnormal coloring
        for i, (category, (min_val, max_val)) in enumerate(categories_data.items()):
            is_normal_category = category == "Normal Healthy Person"
            color = 'green' if is_normal_category else 'lightcoral'
            alpha = 0.7 if category in hr_categories else 0.3
            plt.barh(i, max_val - min_val, left=min_val, height=0.5, 
                    alpha=alpha, color=color)
            plt.text(min_val - 5, i, f"{min_val}", va='center', ha='right')
            plt.text(max_val + 5, i, f"{max_val}", va='center', ha='left')
        
        # Plot vertical line for the detected heart rate
        plt.axvline(x=heart_rate, color='black', linestyle='-', linewidth=2)
        plt.text(heart_rate + 2, len(category_names) - 0.5, f"Heart Rate: {heart_rate:.1f} BPM", 
                 color='black', fontweight='bold')
        
        plt.yticks(y_positions, category_names)
        plt.xlabel('Heart Rate (Beats Per Minute)')
        plt.title(f'Heart Rate Classification - {file_info["file_name"]}')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save figure instead of showing it
        plt.savefig(f"{file_info['file_name']}_heart_rate.png")
        plt.close()
    
    # Create a comprehensive report
    print("\n" + "="*60)
    print(f"COMPREHENSIVE ECG ANALYSIS REPORT - {file_info['file_name']}")
    print("="*60)
    print(f"File analyzed: {file_info['file_name']}")
    print(f"Lead Configuration: {file_info['lead_config']['type']}")
    print(f"Lead Placement Details:")
    for lead, placement in file_info['lead_config']['lead_placement'].items():
        print(f"  - {lead}: {placement}")
    print(f"Total duration: {len(file_info['lead1'])/file_info['fs']:.2f} seconds")
    
    # Add normality status to comprehensive report
    print(f"\nECG Normality Status: {'NORMAL' if normality_report['is_normal'] else 'ABNORMAL'} " +
          f"(Confidence: {normality_report['confidence']:.1f}%)")
    print(f"Risk Level: {normality_report['risk_level']}")
    
    if file_info['heart_rate'] is not None:
        print(f"Heart Rate: {file_info['heart_rate']:.1f} BPM")
        print("\nPossible heart condition categories:")
        for category in file_info['hr_categories']:
            print(f"- {category}")
    else:
        print("Heart rate information not available")
    
    print("\nDSNN Classification Summary:")
    
    dominant_class = file_info['unique_classes'][np.argmax(file_info['counts'])]
    dominant_percentage = (file_info['counts'][np.argmax(file_info['counts'])] / len(file_info['predictions'])) * 100
    
    for cls, count in zip(file_info['unique_classes'], file_info['counts']):
        class_name = class_definitions.get(cls, f"Class {cls}")
        percentage = (count / len(file_info['predictions'])) * 100
        print(f"- {class_name}: {count} segments ({percentage:.2f}%)")
    
    print("\nDominant ECG Pattern:")
    dominant_class_name = class_definitions.get(dominant_class, f"Class {dominant_class}")
    print(f"- {dominant_class_name} ({dominant_percentage:.2f}%)")
    
    print("\nRecommendations:")
    if normality_report['is_normal']:
        print("- ECG appears normal. Continue with regular health monitoring.")
    else:
        print("- Abnormal ECG patterns detected. Consider consulting a healthcare professional.")
        for factor in normality_report['risk_factors']:
            print(f"- {factor}")
    
    print("="*60)

# Save all normality reports to a file for future reference
import json
with open('normality_reports.json', 'w') as f:
    json.dump(normality_reports, f, indent=4)

# ------- PART 3: COMPARATIVE ANALYSIS WITH NORMAL/ABNORMAL FOCUS -------

# Perform comparative analysis if we have processed multiple files
if len(all_file_info) > 1:
    print("\n\n" + "="*70)
    print("COMPARATIVE ANALYSIS ACROSS ALL DATASETS WITH NORMAL/ABNORMAL FOCUS")
    print("="*70)
    
    # Sort datasets consistently to ensure reproducible output
    sorted_indices = sorted(range(len(all_file_info)), key=lambda i: all_file_info[i]['file_name'])
    
    # Reorder all data based on the sorted indices
    dataset_names = [all_file_info[i]['file_name'] for i in sorted_indices]
    normal_percentages = [normality_reports[i]['normal_segments_percentage'] for i in sorted_indices]
    abnormal_percentages = [normality_reports[i]['abnormal_segments_percentage'] for i in sorted_indices]
    risk_levels = [normality_reports[i]['risk_level'] for i in sorted_indices]
    sorted_reports = [normality_reports[i] for i in sorted_indices]
    sorted_file_info = [all_file_info[i] for i in sorted_indices]
    
    # Visualize normality distribution across datasets
    plt.figure(figsize=(12, 8))
    
    # Create stacked bars for normal/abnormal
    bar_width = 0.7
    bars1 = plt.bar(dataset_names, normal_percentages, bar_width, label='Normal ECG', color='green', alpha=0.7)
    bars2 = plt.bar(dataset_names, abnormal_percentages, bar_width, bottom=normal_percentages, 
                   label='Abnormal ECG', color='red', alpha=0.7)
    
    # Add percentage labels
    for bar, percentage in zip(bars1, normal_percentages):
        if percentage > 10:  # Only show label if there's enough space
            plt.text(bar.get_x() + bar.get_width()/2, percentage/2, 
                    f'{percentage:.1f}%', ha='center', va='center', color='white', fontweight='bold')
    
    for bar, norm_pct, abnorm_pct in zip(bars2, normal_percentages, abnormal_percentages):
        if abnorm_pct > 10:  # Only show label if there's enough space
            plt.text(bar.get_x() + bar.get_width()/2, norm_pct + abnorm_pct/2, 
                    f'{abnorm_pct:.1f}%', ha='center', va='center', color='white', fontweight='bold')
    
    plt.xlabel('Dataset')
    plt.ylabel('Percentage of ECG Segments')
    plt.title('Normal vs Abnormal ECG Distribution Across Datasets')
    plt.ylim(0, 100)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    
    # Add risk level as text - adjusted to ensure it's inside the plot
    for i, risk in enumerate(risk_levels):
        color = 'green' if risk == 'Low' else 'orange' if risk == 'Moderate' else 'red'
        # Place text at 95% of max height instead of fixed position
        plt.text(i, 95, f'Risk: {risk}', ha='center', va='bottom', color=color, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust to ensure legend and risk levels are visible
    plt.savefig('ecg_normality_comparison.png', dpi=300, bbox_inches='tight')  # Save with consistent settings
    plt.show()
    
    # Compare dominant abnormality types across datasets
    print("\nComparison of Dominant Abnormality Types:")
    print("-" * 60)
    print(f"{'Dataset':<15} | {'Dominant Abnormality':<30} | {'Prevalence':<10} | {'Risk Level':<12}")
    print("-" * 60)
    
    for i, report in enumerate(sorted_reports):
        if not report['is_normal'] and report['abnormality_types']:
            # Find the most prevalent abnormality
            # Sort by percentage and alphabetically for ties to ensure consistency
            sorted_abnormalities = sorted(report['abnormality_types'].items(), 
                                         key=lambda x: (-x[1]['percentage'], x[0]))
            dominant_abnormality = sorted_abnormalities[0]
            abnormality_name = dominant_abnormality[0]
            percentage = dominant_abnormality[1]['percentage']
            risk = report['risk_level']
            print(f"{dataset_names[i]:<15} | {abnormality_name:<30} | {percentage:.1f}%{' ':>5} | {risk:<12}")
        else:
            print(f"{dataset_names[i]:<15} | {'No significant abnormalities':<30} | {'-':<10} | {report['risk_level']:<12}")
    
    # Compare heart rates across datasets with consistent ordering
    heart_rates = [info.get('heart_rate', None) for info in sorted_file_info]
    valid_indices = []
    valid_heart_rates = []
    
    for i, hr in enumerate(heart_rates):
        if hr is not None:
            valid_indices.append(i)
            valid_heart_rates.append(hr)
    
    valid_datasets = [dataset_names[i] for i in valid_indices]
    
    if valid_heart_rates:
        plt.figure(figsize=(12, 6))
        bars = plt.bar(valid_datasets, valid_heart_rates, color='skyblue')
        
        # Add horizontal lines for normal range
        plt.axhline(y=60, color='green', linestyle='--', alpha=0.7, label='Normal Range (60-100 BPM)')
        plt.axhline(y=100, color='green', linestyle='--', alpha=0.7)
        
        # Use explicit x-coordinates for fill_between
        x_min, x_max = -0.5, len(valid_datasets) - 0.5  # Use plot coordinates
        plt.fill_between([x_min, x_max], 60, 100, color='green', alpha=0.1)
        
        # Add value labels on bars
        for bar, hr in zip(bars, valid_heart_rates):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{hr:.1f}', ha='center', va='bottom')
            
            # Color code the text based on normal/abnormal
            if hr < 60 or hr > 100:
                plt.text(bar.get_x() + bar.get_width()/2., height/2,
                        'Abnormal', ha='center', va='center', color='white', fontweight='bold', rotation=90)
        
        plt.xlabel('Dataset')
        plt.ylabel('Heart Rate (BPM)')
        plt.title('Comparison of Heart Rates Across Datasets')
        plt.legend()
        plt.tight_layout()
        plt.savefig('ecg_heart_rate_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Create a comprehensive summary table with consistent ordering
    print("\nComprehensive ECG Analysis Summary:")
    print("=" * 100)
    headers = ["Dataset", "Normal %", "Abnormal %", "Heart Rate", "Risk Level", "Dominant Pattern", "Key Risk Factors"]
    print(f"{headers[0]:<15} | {headers[1]:<9} | {headers[2]:<10} | {headers[3]:<10} | {headers[4]:<11} | {headers[5]:<20} | {headers[6]}")
    print("=" * 100)
    
    for i, (info, report) in enumerate(zip(sorted_file_info, sorted_reports)):
        # Get dominant pattern
        if 'counts' in info and 'unique_classes' in info and len(info['counts']) > 0:
            dominant_idx = np.argmax(info['counts'])
            dominant_class = info['unique_classes'][dominant_idx]
            dominant_pattern = class_definitions.get(str(dominant_class), f"Class {dominant_class}")
        else:
            dominant_pattern = "Unknown"
        
        # Format heart rate
        heart_rate_str = f"{info.get('heart_rate', 0):.1f}" if info.get('heart_rate') is not None else "N/A"
        
        # Get key risk factor with deterministic selection
        # Sort risk factors alphabetically to ensure consistency
        sorted_risk_factors = sorted(report['risk_factors']) if report['risk_factors'] else ["None identified"]
        key_risk = sorted_risk_factors[0]
        
        # Truncate key risk if too long
        if len(key_risk) > 40:
            key_risk = key_risk[:37] + "..."
        
        print(f"{dataset_names[i]:<15} | {report['normal_segments_percentage']:<9.1f} | {report['abnormal_segments_percentage']:<10.1f} | {heart_rate_str:<10} | {report['risk_level']:<11} | {dominant_pattern:<20} | {key_risk}")
    
    print("=" * 100)

# ------- PART 4: EXPORT RESULTS AND CREATE PATIENT REPORT -------

# Import necessary libraries at the top level
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def export_results_to_txt():
    """Export analysis results to txt files with consistent ordering"""
    
    # Sort reports by file name for consistency
    sorted_reports = sorted(normality_reports, key=lambda x: x['file_name'])
    
    # Export normality reports
    with open('ecg_normality_reports.txt', 'w', newline='') as f:
        # Write header
        f.write('File Name,Is Normal,Confidence,Normal %,Abnormal %,Heart Rate,Risk Level,Risk Factors\n')
        
        # Write data
        for report in sorted_reports:
            # Check if risk_factors exists and handle it safely
            risk_factors = '; '.join(report.get('risk_factors', [])) if report.get('risk_factors') else 'None'
            
            # Safely handle heart rate which might be missing
            heart_rate_str = f"{report.get('heart_rate', 0):.1f}" if report.get('heart_rate') is not None else 'N/A'
            
            # Write data line with proper error handling
            f.write(f"{report['file_name']},{report.get('is_normal', False)},{report.get('confidence', 0):.1f}%,")
            f.write(f"{report.get('normal_segments_percentage', 0):.1f}%,{report.get('abnormal_segments_percentage', 0):.1f}%,")
            f.write(f"{heart_rate_str},")
            f.write(f"{report.get('risk_level', 'Unknown')},{risk_factors}\n")
    
    # Export abnormality details with consistent ordering
    with open('ecg_abnormality_details.txt', 'w', newline='') as f:
        # Write header
        f.write('File Name,Abnormality Type,Count,Percentage\n')
        
        # First gather all entries, then sort for consistent output
        all_entries = []
        for report in sorted_reports:
            # Handle missing abnormality_types
            abnormality_types = report.get('abnormality_types', {})
            for abnormality, stats in abnormality_types.items():
                all_entries.append({
                    'file_name': report['file_name'],
                    'abnormality': abnormality,
                    'count': stats.get('count', 0),
                    'percentage': stats.get('percentage', 0)
                })
        
        # Sort entries for consistent output
        sorted_entries = sorted(all_entries, key=lambda x: (x['file_name'], x['abnormality']))
        
        # Write sorted data
        for entry in sorted_entries:
            f.write(f"{entry['file_name']},{entry['abnormality']},{entry['count']},{entry['percentage']:.1f}%\n")
    
    print("\nResults exported to txt files:")
    print("- ecg_normality_reports.txt")
    print("- ecg_abnormality_details.txt")

def generate_patient_report(file_idx=0):
    """Generate a detailed patient report for a specific dataset"""
    # Validate input
    if not all_file_info or file_idx >= len(all_file_info) or file_idx < 0:
        print(f"Error: File index {file_idx} is out of range or no files have been processed")
        return
    
    try:
        file_info = all_file_info[file_idx]
        report = normality_reports[file_idx]
        
        # Create a deterministic filename based on file info
        pdf_filename = f"patient_ecg_report_{file_info['file_name'].replace(' ', '_')}.pdf"
        doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
        elements = []
        
        styles = getSampleStyleSheet()
        title_style = styles['Title']
        heading_style = styles['Heading1']
        normal_style = styles['Normal']
        
        # Title
        elements.append(Paragraph(f"ECG Analysis Report - Patient ID: {file_info['file_name']}", title_style))
        elements.append(Spacer(1, 12))
        
        # Basic information - safe access to nested dictionaries
        elements.append(Paragraph("Basic Information", heading_style))
        lead_config = file_info.get('lead_config', {})
        lead_type = lead_config.get('type', 'Unknown') if isinstance(lead_config, dict) else 'Unknown'
        
        fs = file_info.get('fs', 0)
        duration = len(file_info.get('lead1', [])) / fs if fs > 0 and 'lead1' in file_info else 0
        
        data = [
            ["Recording Date:", "N/A (Not available in data)"],
            ["Lead Configuration:", lead_type],
            ["Recording Duration:", f"{duration:.2f} seconds"],
            ["Sampling Rate:", f"{fs} Hz"]
        ]
        
        t = Table(data, colWidths=[150, 350])
        t.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey)
        ]))
        elements.append(t)
        elements.append(Spacer(1, 12))
        
        # Analysis results - with safe access to report values
        elements.append(Paragraph("ECG Analysis Results", heading_style))
        
        # Safely get values with defaults
        is_normal = report.get('is_normal', False)
        confidence = report.get('confidence', 0)
        risk_level = report.get('risk_level', 'Unknown')
        normal_count = report.get('normal_segments_count', 0)
        normal_pct = report.get('normal_segments_percentage', 0)
        abnormal_count = report.get('abnormal_segments_count', 0)
        abnormal_pct = report.get('abnormal_segments_percentage', 0)
        
        # Format risk level with color
        risk_color = "green" if risk_level == "Low" else "orange" if risk_level == "Moderate" else "red"
        risk_text = f"<font color='{risk_color}'><b>{risk_level}</b></font>"
        
        # Format normality status with color
        norm_status = "NORMAL" if is_normal else "ABNORMAL"
        norm_color = "green" if is_normal else "red"
        norm_text = f"<font color='{norm_color}'><b>{norm_status}</b></font> (Confidence: {confidence:.1f}%)"
        
        data = [
            ["ECG Classification:", norm_text],
            ["Risk Level:", risk_text],
            ["Normal Segments:", f"{normal_count} ({normal_pct:.1f}%)"],
            ["Abnormal Segments:", f"{abnormal_count} ({abnormal_pct:.1f}%)"]
        ]
        
        # Add heart rate if available
        heart_rate = report.get('heart_rate')
        if heart_rate is not None:
            hr_color = "green" if 60 <= heart_rate <= 100 else "orange"
            hr_text = f"<font color='{hr_color}'><b>{heart_rate:.1f} BPM</b></font>"
            data.append(["Heart Rate:", hr_text])
            
            # Add heart rate categories if available
            hr_categories = report.get('hr_categories', [])
            if hr_categories:
                hr_categories_text = ", ".join(hr_categories)
                data.append(["Heart Rate Categories:", hr_categories_text])
        
        t = Table(data, colWidths=[150, 350])
        t.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey)
        ]))
        elements.append(t)
        elements.append(Spacer(1, 12))
        
        # Abnormality details if any
        abnormality_types = report.get('abnormality_types', {})
        if abnormality_types:
            elements.append(Paragraph("Detected Abnormalities", heading_style))
            abnormality_data = [["Abnormality Type", "Count", "Percentage"]]
            
            # Sort by percentage, then alphabetically for consistent ordering
            sorted_abnormalities = sorted(
                abnormality_types.items(), 
                key=lambda x: (-x[1].get('percentage', 0), x[0])
            )
            
            for abnormality, stats in sorted_abnormalities:
                abnormality_data.append([
                    abnormality,
                    str(stats.get('count', 0)),
                    f"{stats.get('percentage', 0):.1f}%"
                ])
            
            t = Table(abnormality_data, colWidths=[250, 100, 150])
            t.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('ALIGN', (1, 0), (2, -1), 'CENTER')
            ]))
            elements.append(t)
            elements.append(Spacer(1, 12))
        
        # Risk factors - with consistent sorting
        elements.append(Paragraph("Risk Assessment", heading_style))
        risk_factors = sorted(report.get('risk_factors', []))  # Sort for consistency
        if risk_factors:
            risk_data = [["Risk Factors"]]
            for factor in risk_factors:
                risk_data.append([factor])
            
            t = Table(risk_data, colWidths=[500])
            t.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('BACKGROUND', (0, 0), (0, 0), colors.lightgrey)
            ]))
            elements.append(t)
        else:
            elements.append(Paragraph("No significant risk factors identified.", normal_style))
        
        elements.append(Spacer(1, 12))
        
        # Recommendations
        elements.append(Paragraph("Recommendations", heading_style))
        if is_normal:
            elements.append(Paragraph("ECG appears normal. Continue with regular health monitoring.", normal_style))
        else:
            elements.append(Paragraph("Abnormal ECG patterns detected. Consider consulting a healthcare professional for further evaluation.", normal_style))
            
            # Add specific recommendations based on abnormalities - with consistent ordering
            recommendations = []
            
            if "Atrial Fibrillation" in abnormality_types:
                recommendations.append("• Monitor for symptoms such as palpitations, shortness of breath, and fatigue.")
                recommendations.append("• Further evaluation for stroke risk may be recommended.")
            
            if "Ventricular Arrhythmia" in abnormality_types:
                recommendations.append("• Prompt cardiology follow-up is recommended.")
                recommendations.append("• Further testing such as Holter monitoring may be beneficial.")
            
            if "ST Segment Abnormality" in abnormality_types:
                recommendations.append("• Evaluation for possible cardiac ischemia may be warranted.")
            
            # Sort recommendations alphabetically for consistent order
            recommendations.sort()
            for rec in recommendations:
                elements.append(Paragraph(rec, normal_style))
        
        # Disclaimer
        elements.append(Spacer(1, 24))
        disclaimer_style = styles['Italic']
        disclaimer = Paragraph("""Disclaimer: This analysis was generated by an automated system and is intended for research purposes only. 
        It does not constitute medical advice and should not be used as a substitute for professional medical diagnosis. 
        Please consult with a qualified healthcare provider regarding any medical concerns.""", disclaimer_style)
        elements.append(disclaimer)
        
        # Build the PDF
        doc.build(elements)
        print(f"\nPatient report generated: {pdf_filename}")
        
    except Exception as e:
        import traceback
        print(f"Error during report generation: {str(e)}")
        print(traceback.format_exc())  # Print stack trace for better debugging
        print("Report generation requires the reportlab library to be installed.")

# Export results and generate patient report for first dataset
if len(all_file_info) > 0:
    try:
        export_results_to_txt()
        generate_patient_report(0)  # Generate report for first dataset
    except ImportError as e:
        print(f"Error: Missing required libraries for report generation: {e}")
        print("Please install the reportlab library using: pip install reportlab")
    except Exception as e:
        print(f"Unexpected error during report generation: {e}")

print("\nECG analysis completed.")

