import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

metadata_path = '/example-path/metadata.csv'
metadata = pd.read_csv(metadata_path)

data_directory = '/example-path/example-directory/'

conditions = ["english-english", "english-spanish", "spanish-spanish", "spanish-english"]
condition_labels = ["English to English", "English to Spanish", "Spanish to Spanish", "Spanish to English"]

sfreq = 100
montage = mne.channels.make_standard_montage('standard_1020')

for _, row in metadata.iterrows():
    participant_number = int(row['participant'])
    speaks_spanish = row['spanish'] == 1

    file_paths = [
        os.path.join(data_directory, f"example_{condition}_translation_{participant_number}.csv")
        for condition in conditions
    ]
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"Participant {participant_number} - {'Speaks Spanish' if speaks_spanish else 'Does Not Speak Spanish'}", fontsize=16)

    for i, (file_path, condition_label) in enumerate(zip(file_paths, condition_labels)):
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
            ch_names = data.columns.tolist()
            info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
            info.set_montage(montage)
            sample_data = data.iloc[0].values
            evoked_array = mne.EvokedArray(sample_data[:, np.newaxis], info)
            mne.viz.plot_topomap(evoked_array.data[:, 0], evoked_array.info, axes=axes[i],
                                 show=False, contours=5, cmap='viridis')
            axes[i].set_title(f"{condition_label}")
        else:
            axes[i].text(0.5, 0.5, 'File not available', ha='center', va='center', fontsize=12, color='gray')
            axes[i].set_title(f"{condition_label}")
            axes[i].set_xticks([])
            axes[i].set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    save_directory = '/example-path/SandiaMapImages/'
    fig.savefig(os.path.join(save_directory, f"Participant_{participant_number}_topomaps.png"))
