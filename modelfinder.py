import pandas as pd
import numpy as np
import os
import glob

data_directory = '/example-path/example-directory/'
metadata_path = '/example-path/metadata.csv'

metadata = pd.read_csv(metadata_path)
participants = range(1, 41)
target_time_point = 122
min_threshold_range = np.arange(0, 5, 0.1)
max_threshold_range = np.arange(0, 5, 0.1)

def calculate_average_voltage_at_time_point(file_path, time_point, electrode_index):
    df = pd.read_csv(file_path)
    if time_point >= df.shape[0] or electrode_index >= df.shape[1]:
        return None
    return df.iloc[time_point, electrode_index]

def classify_spanish_speaker(abs_diff, min_threshold, max_threshold):
    return 1 if min_threshold <= abs_diff <= max_threshold else 0

def evaluate_accuracy(min_threshold, max_threshold, abs_diffs):
    correct_predictions = 0
    total_predictions = 0
    spanish_correct = 0
    for participant, abs_diff in abs_diffs:
        actual = metadata.loc[metadata['participant'] == participant, 'spanish'].values[0]
        prediction = classify_spanish_speaker(abs_diff, min_threshold, max_threshold)
        if prediction == actual:
            correct_predictions += 1
        if actual == 1 and prediction == 1:
            spanish_correct += 1
        total_predictions += 1
    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    return accuracy, spanish_correct

high_accuracy_results = []
total_iterations = len(range(30)) * len(min_threshold_range) * len(max_threshold_range)
current_iteration = 0
last_reported_percent = 0

for electrode_index in range(30):
    absolute_differences = []
    for participant in participants:
        related_files = glob.glob(os.path.join(data_directory, f"*spanish-english_translation_{participant}.csv"))
        unrelated_files = glob.glob(os.path.join(data_directory, f"*english-english_translation_{participant}.csv"))
        related_values = [
            calculate_average_voltage_at_time_point(file_path, target_time_point, electrode_index)
            for file_path in related_files if os.path.exists(file_path)
        ]
        related_values = [val for val in related_values if val is not None]
        unrelated_values = [
            calculate_average_voltage_at_time_point(file_path, target_time_point, electrode_index)
            for file_path in unrelated_files if os.path.exists(file_path)
        ]
        unrelated_values = [val for val in unrelated_values if val is not None]
        if related_values and unrelated_values:
            avg_related = np.mean(related_values)
            avg_unrelated = np.mean(unrelated_values)
            abs_diff = abs(avg_related - avg_unrelated)
            absolute_differences.append((participant, abs_diff))
    for min_threshold in min_threshold_range:
        for max_threshold in max_threshold_range:
            if min_threshold < max_threshold:
                accuracy, spanish_correct = evaluate_accuracy(min_threshold, max_threshold, absolute_differences)
                if accuracy > 75 and spanish_correct >= 7:
                    high_accuracy_results.append({
                        "Electrode": electrode_index,
                        "Min Threshold": min_threshold,
                        "Max Threshold": max_threshold,
                        "Accuracy": accuracy,
                        "Spanish Correct": spanish_correct
                    })
                    print(f"Electrode {electrode_index}, Window ({min_threshold}, {max_threshold}), "
                          f"Accuracy: {accuracy:.2f}%, Spanish Correct: {spanish_correct}")
                current_iteration += 1
                progress_percent = (current_iteration / total_iterations) * 100
                if int(progress_percent) > last_reported_percent:
                    print(f"Progress: {int(progress_percent)}% complete")
                    last_reported_percent = int(progress_percent)

print("\nHigh-Accuracy Results (Accuracy > 85% and Spanish Correct >= 8):")
for result in high_accuracy_results:
    print(result)
