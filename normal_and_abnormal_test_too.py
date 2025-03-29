import numpy as np
import tensorflow as tf
import torch
import pyedflib
import matplotlib.pyplot as plt
import os
import random
import wfdb  # For handling QRS annotations

# Add these at the beginning of your script
os.environ['PYTHONHASHSEED'] = '42'
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

from dsnn_example import DSNN, DSNNSystem


# ------- PART 1: DATA PREPARATION FOR MULTIPLE DATASETS -------

# 1. Function to process a single file
def process_single_file(base_path, file_name, using_sliding_window=False):
    edf_path = os.path.join(base_path, file_name + ".edf")
    qrs_path = os.path.join(base_path, file_name)  # QRS file path without extension
    
    print(f"\nProcessing file: {file_name}")
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
        f.close()
        return None
        
    lead1 = f.readSignal(0)  # First channel
    lead2 = f.readSignal(1)  # Second channel
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
    if using_qrs and len(r_peaks) > 0:
        print("Extracting segments centered around R-peaks...")
        segments = extract_segments_around_rpeaks(lead1, lead2, r_peaks)
    else:
        print("Extracting segments using sliding window...")
        segments = extract_segments_sliding_window(lead1, lead2)
    
    print(f"Extracted {len(segments)} segments of length 24 from the ECG data")
    
    # Calculate heart rate if QRS annotations are available
    heart_rate = None
    hr_categories = []
    
    if using_qrs and len(r_peaks) > 1:
        heart_rate = calculate_heart_rate(r_peaks, fs)
        if heart_rate is not None:
            print(f"\nCalculated heart rate: {heart_rate:.1f} BPM")
            hr_categories = classify_heart_rate(heart_rate)
            print("Possible categories based on heart rate:")
            for category in hr_categories:
                print(f"- {category}")
    else:
        print("\nHeart rate calculation requires QRS annotations, which are not available")
    
    # Close the EDF file
    f.close()
    
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
        "lead1": lead1,
        "lead2": lead2
    }
    
    return file_info

# New function to determine lead configuration based on signal labels
def determine_lead_configuration(signal_labels):
    """
    Determine the lead configuration based on channel labels
    Returns a dictionary with lead configuration and placement information
    """
    # Convert labels to lowercase for case-insensitive matching
    lower_labels = [label.lower() if isinstance(label, str) else "" for label in signal_labels]
    
    # Standard 12-lead ECG configuration detection
    standard_leads = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
    found_leads = [lead for lead in standard_leads if any(lead in label for label in lower_labels)]
    
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
        config = {
            "type": "Custom ECG configuration",
            "description": f"Non-standard lead configuration: {', '.join(signal_labels[:5])}",
            "lead_placement": {
                signal_labels[0]: "Primary recording lead",
                signal_labels[1]: "Secondary recording lead"
            }
        }
    
    return config

# 2. Extract segments around R-peaks or using sliding window
def extract_segments_around_rpeaks(lead1, lead2, r_peaks, segment_length=24, offset=12):
    """Extract segments centered around R-peaks"""
    segments = []
    
    for peak in r_peaks:
        # Calculate segment boundaries
        start = peak - offset
        end = start + segment_length
        
        # Skip if segment would go out of bounds
        if start < 0 or end > len(lead1):
            continue
            
        # Extract segment from both leads
        segment1 = lead1[start:end]
        segment2 = lead2[start:end]
        
        # Stack the two leads
        segment = np.stack([segment1, segment2])
        segments.append(segment)
    
    return np.array(segments)

def extract_segments_sliding_window(lead1, lead2, segment_length=24, stride=12):
    """Extract segments using sliding window"""
    segments = []
    
    # Make sure both leads have the same length
    min_length = min(len(lead1), len(lead2))
    lead1 = lead1[:min_length]
    lead2 = lead2[:min_length]
    
    # Extract segments with a specified stride
    for i in range(0, min_length - segment_length, stride):
        segment1 = lead1[i:i+segment_length]
        segment2 = lead2[i:i+segment_length]
        
        # Stack the two leads
        segment = np.stack([segment1, segment2])
        segments.append(segment)
    
    return np.array(segments)

# 3. Define heart rate calculation and classification functions
def calculate_heart_rate(r_peaks, fs):
    """Calculate heart rate in BPM from R-peak locations"""
    if len(r_peaks) < 2:
        return None
    
    # Calculate RR intervals in samples
    rr_intervals = np.diff(r_peaks)
    
    # Convert to seconds
    rr_seconds = rr_intervals / fs
    
    # Calculate instantaneous heart rates
    inst_hr = 60 / rr_seconds
    
    # Return median heart rate (more robust than mean)
    return np.median(inst_hr)

def classify_heart_rate(bpm):
    """Classify heart rate based on the provided categories"""
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

# NEW FUNCTIONS FOR NORMAL/ABNORMAL CLASSIFICATION

def is_normal_ecg(predictions):
    """
    Determine if an ECG is normal based on DSNN predictions
    Returns a tuple (is_normal, confidence)
    """
    # Class 0 represents Normal Sinus Rhythm
    normal_class = 0
    
    # Count occurrences of normal rhythm
    if normal_class in predictions:
        normal_count = np.sum(predictions == normal_class)
        total_segments = len(predictions)
        normal_percentage = (normal_count / total_segments) * 100
        
        # If more than 80% of segments are classified as normal, consider the ECG normal
        is_normal = normal_percentage >= 80
        confidence = normal_percentage if is_normal else (100 - normal_percentage)
        
        return is_normal, confidence
    else:
        # If normal class not found at all, it's definitely abnormal
        return False, 100.0

def classify_abnormality_type(predictions, class_definitions):
    """
    Classify the type of abnormality based on DSNN predictions
    Returns a dictionary with abnormality types and their prevalence
    """
    # Remove normal class from analysis
    abnormal_predictions = predictions[predictions != 0]
    
    if len(abnormal_predictions) == 0:
        return {}
    
    # Count occurrences of each abnormal class
    abnormality_types = {}
    unique_classes, counts = np.unique(abnormal_predictions, return_counts=True)
    
    for cls, count in zip(unique_classes, counts):
        percentage = (count / len(abnormal_predictions)) * 100
        class_name = class_definitions.get(cls, f"Unknown Class {cls}")
        abnormality_types[class_name] = {
            "count": int(count),
            "percentage": float(percentage)
        }
    
    return abnormality_types

def visualize_normal_abnormal_segments(segments, predictions, num_samples=3):
    """
    Visualize example normal and abnormal ECG segments
    """
    # Get indices of normal and abnormal segments
    normal_indices = np.where(predictions == 0)[0]
    abnormal_indices = np.where(predictions != 0)[0]
    
    # If we don't have both normal and abnormal segments, return
    if len(normal_indices) == 0 or len(abnormal_indices) == 0:
        print("Could not visualize normal/abnormal segments: missing either normal or abnormal examples")
        return
    
    # Select random samples
    np.random.seed(42)  # For reproducibility
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
    """
    predictions = file_info['predictions']
    is_normal, confidence = is_normal_ecg(predictions)
    
    # Prepare the report
    report = {
        "file_name": file_info['file_name'],
        "is_normal": is_normal,
        "confidence": confidence,
        "normal_segments_count": np.sum(predictions == 0),
        "normal_segments_percentage": (np.sum(predictions == 0) / len(predictions)) * 100,
        "abnormal_segments_count": np.sum(predictions != 0),
        "abnormal_segments_percentage": (np.sum(predictions != 0) / len(predictions)) * 100,
        "abnormality_types": classify_abnormality_type(predictions, class_definitions),
        "heart_rate": file_info.get('heart_rate', None),
        "hr_categories": file_info.get('hr_categories', [])
    }
    
    # Add risk assessment based on abnormality types and heart rate
    risk_level = "Low"
    risk_factors = []
    
    # Assess risk based on abnormality prevalence
    if report["abnormal_segments_percentage"] > 50:
        risk_level = "High"
        risk_factors.append(f"Majority of ECG segments ({report['abnormal_segments_percentage']:.1f}%) show abnormalities")
    elif report["abnormal_segments_percentage"] > 20:
        risk_level = "Moderate"
        risk_factors.append(f"Significant proportion of ECG segments ({report['abnormal_segments_percentage']:.1f}%) show abnormalities")
    
    # Assess risk based on heart rate
    if file_info.get('heart_rate') is not None:
        hr = file_info['heart_rate']
        if hr > 100:
            risk_factors.append(f"Elevated heart rate ({hr:.1f} BPM)")
            if risk_level == "Low":
                risk_level = "Moderate"
        elif hr < 60:
            risk_factors.append(f"Low heart rate ({hr:.1f} BPM)")
            if risk_level == "Low":
                risk_level = "Moderate"
    
    # Assess risk based on abnormality types
    high_risk_conditions = ["Ventricular Arrhythmia", "ST Segment Abnormality"]
    for condition in high_risk_conditions:
        if condition in report["abnormality_types"] and report["abnormality_types"][condition]["percentage"] > 10:
            risk_factors.append(f"Presence of {condition} ({report['abnormality_types'][condition]['percentage']:.1f}%)")
            risk_level = "High"
    
    report["risk_level"] = risk_level
    report["risk_factors"] = risk_factors
    
    return report

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
    plt.show()
    
    # Process segments in batches
    print("\nRunning DSNN algorithm on the data...")
    batch_size = 32
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
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
        plt.show()
    
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
    
    plt.show()
    
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
        plt.show()
    
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

# ------- PART 3: COMPARATIVE ANALYSIS WITH NORMAL/ABNORMAL FOCUS -------

# Perform comparative analysis if we have processed multiple files
if len(all_file_info) > 1:
    print("\n\n" + "="*70)
    print("COMPARATIVE ANALYSIS ACROSS ALL DATASETS WITH NORMAL/ABNORMAL FOCUS")
    print("="*70)
    
# Compare normality across datasets
    dataset_names = [info['file_name'] for info in all_file_info]
    normal_percentages = [report['normal_segments_percentage'] for report in normality_reports]
    abnormal_percentages = [report['abnormal_segments_percentage'] for report in normality_reports]
    risk_levels = [report['risk_level'] for report in normality_reports]
    
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
    
    # Add risk level as text
    for i, risk in enumerate(risk_levels):
        color = 'green' if risk == 'Low' else 'orange' if risk == 'Moderate' else 'red'
        plt.text(i, 102, f'Risk: {risk}', ha='center', va='bottom', color=color, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Compare dominant abnormality types across datasets
    print("\nComparison of Dominant Abnormality Types:")
    print("-" * 60)
    print(f"{'Dataset':<10} | {'Dominant Abnormality':<30} | {'Prevalence':<10} | {'Risk Level':<12}")
    print("-" * 60)
    
    for i, report in enumerate(normality_reports):
        if not report['is_normal'] and report['abnormality_types']:
            # Find the most prevalent abnormality
            dominant_abnormality = max(report['abnormality_types'].items(), 
                                      key=lambda x: x[1]['percentage'])
            abnormality_name = dominant_abnormality[0]
            percentage = dominant_abnormality[1]['percentage']
            risk = report['risk_level']
            print(f"{dataset_names[i]:<10} | {abnormality_name:<30} | {percentage:.1f}%{' ':>5} | {risk:<12}")
        else:
            print(f"{dataset_names[i]:<10} | {'No significant abnormalities':<30} | {'-':<10} | {report['risk_level']:<12}")
    
    # Compare heart rates across datasets
    heart_rates = [info.get('heart_rate', None) for info in all_file_info]
    valid_datasets = []
    valid_heart_rates = []
    
    for name, hr in zip(dataset_names, heart_rates):
        if hr is not None:
            valid_datasets.append(name)
            valid_heart_rates.append(hr)
    
    if valid_heart_rates:
        plt.figure(figsize=(12, 6))
        bars = plt.bar(valid_datasets, valid_heart_rates, color='skyblue')
        
        # Add horizontal lines for normal range
        plt.axhline(y=60, color='green', linestyle='--', alpha=0.7, label='Normal Range (60-100 BPM)')
        plt.axhline(y=100, color='green', linestyle='--', alpha=0.7)
        plt.fill_between([min(valid_datasets), max(valid_datasets)], 60, 100, color='green', alpha=0.1)
        
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
        plt.show()
    
    # Create a comprehensive summary table
    print("\nComprehensive ECG Analysis Summary:")
    print("=" * 100)
    headers = ["Dataset", "Normal %", "Abnormal %", "Heart Rate", "Risk Level", "Dominant Pattern", "Key Risk Factors"]
    print(f"{headers[0]:<10} | {headers[1]:<9} | {headers[2]:<10} | {headers[3]:<10} | {headers[4]:<11} | {headers[5]:<20} | {headers[6]}")
    print("=" * 100)
    
    for i, (info, report) in enumerate(zip(all_file_info, normality_reports)):
        # Get dominant pattern
        dominant_idx = np.argmax(info['counts'])
        dominant_class = info['unique_classes'][dominant_idx]
        dominant_pattern = class_definitions.get(dominant_class, f"Class {dominant_class}")
        
        # Format heart rate
        heart_rate_str = f"{info.get('heart_rate', 0):.1f}" if info.get('heart_rate') is not None else "N/A"
        
        # Get key risk factor (first one or empty)
        key_risk = report['risk_factors'][0] if report['risk_factors'] else "None identified"
        
        # Truncate key risk if too long
        if len(key_risk) > 40:
            key_risk = key_risk[:37] + "..."
        
        print(f"{dataset_names[i]:<10} | {report['normal_segments_percentage']:<9.1f} | {report['abnormal_segments_percentage']:<10.1f} | {heart_rate_str:<10} | {report['risk_level']:<11} | {dominant_pattern:<20} | {key_risk}")
    
    print("=" * 100)

# ------- PART 4: EXPORT RESULTS AND CREATE PATIENT REPORT -------

def export_results_to_txt():
    """Export analysis results to txt files"""
    import reportlab
    
    # Export normality reports
    with open('ecg_normality_reports.txt', 'w', newline='') as f:
        # Write header
        f.write('File Name,Is Normal,Confidence,Normal %,Abnormal %,Heart Rate,Risk Level,Risk Factors\n')
        
        # Write data
        for report in normality_reports:
            risk_factors = '; '.join(report['risk_factors']) if report['risk_factors'] else 'None'
            heart_rate_str = f"{report['heart_rate']:.1f}" if report['heart_rate'] is not None else 'N/A'
            
            f.write(f"{report['file_name']},{report['is_normal']},{report['confidence']:.1f}%,")
            f.write(f"{report['normal_segments_percentage']:.1f}%,{report['abnormal_segments_percentage']:.1f}%,")
            f.write(f"{heart_rate_str},")
            f.write(f"{report['risk_level']},{risk_factors}\n")
    
    # Export abnormality details
    with open('ecg_abnormality_details.txt', 'w', newline='') as f:
        # Write header
        f.write('File Name,Abnormality Type,Count,Percentage\n')
        
        # Write data
        for report in normality_reports:
            for abnormality, stats in report['abnormality_types'].items():
                f.write(f"{report['file_name']},{abnormality},{stats['count']},{stats['percentage']:.1f}%\n")
    
    print("\nResults exported to txt files:")
    print("- ecg_normality_reports.txt")
    print("- ecg_abnormality_details.txt")

def generate_patient_report(file_idx=0):
    """Generate a detailed patient report for a specific dataset"""
    if file_idx >= len(all_file_info):
        print(f"Error: File index {file_idx} is out of range")
        return
    
    file_info = all_file_info[file_idx]
    report = normality_reports[file_idx]
    
    # Create the report
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    
    pdf_filename = f"patient_ecg_report_{file_info['file_name']}.pdf"
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
    elements = []
    
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    heading_style = styles['Heading1']
    normal_style = styles['Normal']
    
    # Title
    elements.append(Paragraph(f"ECG Analysis Report - Patient ID: {file_info['file_name']}", title_style))
    elements.append(Spacer(1, 12))
    
    # Basic information
    elements.append(Paragraph("Basic Information", heading_style))
    data = [
        ["Recording Date:", "N/A (Not available in data)"],
        ["Lead Configuration:", file_info['lead_config']['type']],
        ["Recording Duration:", f"{len(file_info['lead1'])/file_info['fs']:.2f} seconds"],
        ["Sampling Rate:", f"{file_info['fs']} Hz"]
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
    
    # Analysis results
    elements.append(Paragraph("ECG Analysis Results", heading_style))
    
    # Format risk level with color
    risk_color = "green" if report['risk_level'] == "Low" else "orange" if report['risk_level'] == "Moderate" else "red"
    risk_text = f"<font color='{risk_color}'><b>{report['risk_level']}</b></font>"
    
    # Format normality status with color
    norm_status = "NORMAL" if report['is_normal'] else "ABNORMAL"
    norm_color = "green" if report['is_normal'] else "red"
    norm_text = f"<font color='{norm_color}'><b>{norm_status}</b></font> (Confidence: {report['confidence']:.1f}%)"
    
    data = [
        ["ECG Classification:", norm_text],
        ["Risk Level:", risk_text],
        ["Normal Segments:", f"{report['normal_segments_count']} ({report['normal_segments_percentage']:.1f}%)"],
        ["Abnormal Segments:", f"{report['abnormal_segments_count']} ({report['abnormal_segments_percentage']:.1f}%)"]
    ]
    
    # Add heart rate if available
    if report['heart_rate'] is not None:
        hr_color = "green" if 60 <= report['heart_rate'] <= 100 else "orange"
        hr_text = f"<font color='{hr_color}'><b>{report['heart_rate']:.1f} BPM</b></font>"
        data.append(["Heart Rate:", hr_text])
        
        # Add heart rate categories if available
        if report['hr_categories']:
            hr_categories_text = ", ".join(report['hr_categories'])
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
    if report['abnormality_types']:
        elements.append(Paragraph("Detected Abnormalities", heading_style))
        abnormality_data = [["Abnormality Type", "Count", "Percentage"]]
        
        for abnormality, stats in sorted(report['abnormality_types'].items(), 
                                          key=lambda x: x[1]['percentage'], reverse=True):
            abnormality_data.append([
                abnormality,
                str(stats['count']),
                f"{stats['percentage']:.1f}%"
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
    
    # Risk factors
    elements.append(Paragraph("Risk Assessment", heading_style))
    if report['risk_factors']:
        risk_data = [["Risk Factors"]]
        for factor in report['risk_factors']:
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
    if report['is_normal']:
        elements.append(Paragraph("ECG appears normal. Continue with regular health monitoring.", normal_style))
    else:
        elements.append(Paragraph("Abnormal ECG patterns detected. Consider consulting a healthcare professional for further evaluation.", normal_style))
        
        # Add specific recommendations based on abnormalities
        if "Atrial Fibrillation" in report['abnormality_types']:
            elements.append(Paragraph("• Monitor for symptoms such as palpitations, shortness of breath, and fatigue.", normal_style))
            elements.append(Paragraph("• Further evaluation for stroke risk may be recommended.", normal_style))
        
        if "Ventricular Arrhythmia" in report['abnormality_types']:
            elements.append(Paragraph("• Prompt cardiology follow-up is recommended.", normal_style))
            elements.append(Paragraph("• Further testing such as Holter monitoring may be beneficial.", normal_style))
        
        if "ST Segment Abnormality" in report['abnormality_types']:
            elements.append(Paragraph("• Evaluation for possible cardiac ischemia may be warranted.", normal_style))
    
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

# Export results and generate patient report for first dataset
if len(all_file_info) > 0:
    try:
        export_results_to_txt()
        generate_patient_report(0)  # Generate report for first dataset
    except Exception as e:
        print(f"Error during report generation: {e}")
        print("Report generation requires additional libraries: reportlab")

print("\nECG analysis completed.")
