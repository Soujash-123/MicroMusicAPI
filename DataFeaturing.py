import os
import librosa
import pandas as pd

# Function to extract features from a single audio file
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

# Function to generate dataset from file path or list of file paths
def generate_dataset(file_paths):
    if isinstance(file_paths, str):
        file_paths = [file_paths]  # Convert single file path to list

    all_features = []
    for file_path in file_paths:
        print(f"Parsing file: {file_path}")
        features = extract_features(file_path)
        if features:
            all_features.append(features)
    return pd.DataFrame(all_features)

def generate_dataset_from_file(file_name):
    all_features = []
    print(f"Parsing file: {file_name}")
    features = extract_features(file_name)
    if features:
        all_features.append(features)
    return pd.DataFrame(all_features)
# Example usage:
if __name__ == "__main__":
    # Example usage with a single file
    '''single_file_path = './musical/audio_file.mp3'
    df_single = generate_dataset(single_file_path)
    print(df_single.head())'''

    # Example usage with multiple files
    '''folder_path = './musical'  # Or any other folder containing audio files
    file_paths = [os.path.join(folder_path, file_name) for file_name in os.listdir(folder_path) if file_name.endswith('.mp3')]
    df_multiple = generate_dataset(file_paths)
    print(df_multiple.head())

    # Save DataFrame to CSV if needed
    output_csv = 'audio_features.csv'
    df_multiple.to_csv(output_csv, index=False)
    print(f"Features extracted from {len(df_multiple)} audio files and saved to {output_csv}.")
    '''
    output_csv = "new.csv"
    df_multiple = generate_dataset_from_file("./musical/Lil Tecca, Juice WRLD - Ransom (Clean - Lyrics).mp3")
    print(df_multiple)    
