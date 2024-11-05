import pandas as pd
import numpy as np
import glob
import os

metadata_path = '/example-path/metadata.csv'
metadata = pd.read_csv(metadata_path)

data_directory = '/example-path/example-directory/'
time_index = 122
min_threshold = 1.3
max_threshold = 2.5

correct_predictions = 0
total_tested = 0
results = []

def threshold_model(value):
    return 1 if min_threshold <= value <= max_threshold else 0

for _, row in metadata.iterrows():
    participant_number = int(row['participant'])
    speaks_spanish = row['spanish']

    spanish_english_files = glob.glob(os.path.join(data_directory, f"*spanish-english_translation_{participant_number}.csv"))
    english_english_files = glob.glob(os.path.join(data_directory, f"*english-english_translation_{participant_number}.csv"))

    spanish_english_values = []
    english_english_values = []

    for file_path in spanish_english_files:
        df = pd.read_csv(file_path)
        if time_index < df.shape[0]:
            spanish_english_values.append(df.iloc[time_index, 7])

    for file_path in english_english_files:
        df = pd.read_csv(file_path)
        if time_index < df.shape[0]:
            english_english_values.append(df.iloc[time_index, 7])

    if spanish_english_values and english_english_values:
        avg_spanish_english = np.mean(spanish_english_values)
        avg_english_english = np.mean(english_english_values)
        abs_diff = abs(avg_spanish_english - avg_english_english)

        prediction = threshold_model(abs_diff)
        is_correct = prediction == speaks_spanish
        if is_correct:
            correct_predictions += 1
        total_tested += 1

        results.append({
            "Participant": participant_number,
            "Actual": speaks_spanish,
            "Absolute Difference": abs_diff,
            "Prediction": prediction,
            "Correct": is_correct
        })

for result in results:
    print(f"Participant {result['Participant']}: "
          f"Actual = {result['Actual']}, "
          f"Absolute Difference = {result['Absolute Difference']:.3f}, "
          f"Prediction = {result['Prediction']}, "
          f"Correct = {result['Correct']}")

print(f"\nCorrect Predictions: {correct_predictions}")
print(f"Total Tested: {total_tested}")
if total_tested > 0:
    accuracy = (correct_predictions / total_tested) * 100
    print(f"Accuracy: {accuracy:.2f}%")
else:
    print("No data to test.")
