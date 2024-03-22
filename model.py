import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import DataFeaturing as dfp
import numpy as np

def predictUsingModel(df2):
    # Load the dataset
    df = pd.read_csv('audio_features.csv')
    
    # Select features (X)
    X = df[['duration', 'tempo', 'onset_strength', 'chroma_stft_mean', 'chroma_stft_std', 'rms_mean', 'rms_std',
            'spectral_centroid_mean', 'spectral_centroid_std', 'spectral_bandwidth_mean', 'spectral_bandwidth_std',
            'spectral_rolloff_mean', 'spectral_rolloff_std', 'zero_crossing_rate_mean', 'zero_crossing_rate_std',
            'mfcc_mean', 'mfcc_std', 'tonnetz_mean', 'tonnetz_std', 'Beats.Per.Minute', 'Loudness..dB..',
            'Length.', 'Acousticness..', 'Speechiness.']]
    
    # Target variables
    Y_energy = df['Energy']
    Y_danceability = df['Danceability']
    Y_liveness = df['Liveness']
    Y_valence = df['Valence.']
    Y_length = df['Length.']
    Y_acousticness = df['Acousticness..']
    Y_speechiness = df['Speechiness.']
    Y_popularity = df['Popularity']
    
    # Split data into training and testing sets for each target variable
    X_train_energy, X_test_energy, Y_train_energy, Y_test_energy = train_test_split(X, Y_energy, test_size=0.2, random_state=42)
    X_train_danceability, X_test_danceability, Y_train_danceability, Y_test_danceability = train_test_split(X, Y_danceability, test_size=0.2, random_state=42)
    X_train_liveness, X_test_liveness, Y_train_liveness, Y_test_liveness = train_test_split(X, Y_liveness, test_size=0.2, random_state=42)
    X_train_valence, X_test_valence, Y_train_valence, Y_test_valence = train_test_split(X, Y_valence, test_size=0.2, random_state=42)
    X_train_length, X_test_length, Y_train_length, Y_test_length = train_test_split(X, Y_length, test_size=0.2, random_state=42)
    X_train_acousticness, X_test_acousticness, Y_train_acousticness, Y_test_acousticness = train_test_split(X, Y_acousticness, test_size=0.2, random_state=42)
    X_train_speechiness, X_test_speechiness, Y_train_speechiness, Y_test_speechiness = train_test_split(X, Y_speechiness, test_size=0.2, random_state=42)
    X_train_popularity, X_test_popularity, Y_train_popularity, Y_test_popularity = train_test_split(X, Y_popularity, test_size=0.2, random_state=42)
    
    # Initialize and train machine learning models (Random Forest Regressor) for each target variable
    model_energy = RandomForestRegressor()
    model_energy.fit(X_train_energy, Y_train_energy)
    
    model_danceability = RandomForestRegressor()
    model_danceability.fit(X_train_danceability, Y_train_danceability)
    
    model_liveness = RandomForestRegressor()
    model_liveness.fit(X_train_liveness, Y_train_liveness)
    
    model_valence = RandomForestRegressor()
    model_valence.fit(X_train_valence, Y_train_valence)
    
    model_length = RandomForestRegressor()
    model_length.fit(X_train_length, Y_train_length)
    
    model_acousticness = RandomForestRegressor()
    model_acousticness.fit(X_train_acousticness, Y_train_acousticness)
    
    model_speechiness = RandomForestRegressor()
    model_speechiness.fit(X_train_speechiness, Y_train_speechiness)
    
    model_popularity = RandomForestRegressor()
    model_popularity.fit(X_train_popularity, Y_train_popularity)
    
    # Make predictions
    energy_predictions = model_energy.predict(X_test_energy)
    danceability_predictions = model_danceability.predict(X_test_danceability)
    liveness_predictions = model_liveness.predict(X_test_liveness)
    valence_predictions = model_valence.predict(X_test_valence)
    length_predictions = model_length.predict(X_test_length)
    acousticness_predictions = model_acousticness.predict(X_test_acousticness)
    speechiness_predictions = model_speechiness.predict(X_test_speechiness)
    popularity_predictions = model_popularity.predict(X_test_popularity)
    
    # Calculate average predictions
    avg_energy = np.mean(energy_predictions)
    avg_danceability = np.mean(danceability_predictions)
    avg_liveness = np.mean(liveness_predictions)
    avg_valence = np.mean(valence_predictions)
    avg_length = np.mean(length_predictions)
    avg_acousticness = np.mean(acousticness_predictions)
    avg_speechiness = np.mean(speechiness_predictions)
    avg_popularity = np.mean(popularity_predictions)
    
    # Return average predictions
    return {
        "Energy": avg_energy,
        "Danceability": avg_danceability,
        "Liveness": avg_liveness,
        "Valence": avg_valence,
        "Length": avg_length,
        "Acousticness": avg_acousticness,
        "Speechiness": avg_speechiness,
        "Popularity": avg_popularity
    }

if __name__ == '__main__':
    song = r"C:\Users\SOUJASH\Desktop\MusicAnalyser\musical\The Chainsmokers, Bebe Rexha - Call You Mine (Lyrics).mp3"
    df2 = dfp.generate_dataset_from_file(song)
    print(predictUsingModel(df2))
