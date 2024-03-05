import os
import librosa
import csv

# Path to the directory containing the MP3 files
folder_path = './musical'

# Function to extract features from an audio file
def extract_features(file_path):
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=None)

        # Extract features
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        # Separate harmonic and percussive components
        y_harmonic, y_percussive = librosa.effects.hpss(y)

        features = {
            'file_name': os.path.basename(file_path),
            'duration': librosa.get_duration(y=y, sr=sr),
            'tempo': tempo,
            'onset_strength': onset_env.mean(),
            'chroma_stft_mean': chroma_stft.mean(),
            'chroma_stft_std': chroma_stft.std(),
            'rms_mean': rms.mean(),
            'rms_std': rms.std(),
            'spectral_centroid_mean': spec_cent.mean(),
            'spectral_centroid_std': spec_cent.std(),
            'spectral_bandwidth_mean': spec_bw.mean(),
            'spectral_bandwidth_std': spec_bw.std(),
            'spectral_rolloff_mean': rolloff.mean(),
            'spectral_rolloff_std': rolloff.std(),
            'zero_crossing_rate_mean': zcr.mean(),
            'zero_crossing_rate_std': zcr.std(),
            'mfcc_mean': mfcc.mean(),
            'mfcc_std': mfcc.std(),
            'tonnetz_mean': tonnetz.mean(),
            'tonnetz_std': tonnetz.std()
            # Add more features as needed
        }
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

# List to store all feature dictionaries
all_features = []

# Iterate through all files in the directory
for file_name in os.listdir(folder_path):
    if file_name.endswith('.mp3'):
        file_path = os.path.join(folder_path, file_name)
        print(f"Parsing file: {file_name}")
        features = extract_features(file_path)
        if features:
            all_features.append(features)

# Path to save the CSV file
output_csv = 'audio_features.csv'

# Write features to CSV file
with open(output_csv, 'w', newline='') as csvfile:
    fieldnames = all_features[0].keys() if all_features else []
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for feature_dict in all_features:
        writer.writerow(feature_dict)

print(f"Features extracted from {len(all_features)} audio files and saved to {output_csv}.")
