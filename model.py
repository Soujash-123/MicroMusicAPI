import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import DataFeaturing as dfp



 __author___ = "Soujash Banerjee"
___organization___= "Institute of Engineering and Management, Salt Lake"
def get_metrics(y_true, y_pred, attribute_name):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    metrics = {
        "attribute_name": attribute_name,
        "RMSE": rmse,
        "MAE": mae,
        "R-squared": r2
    }
    return metrics

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
    metrics_list = [
        get_metrics(Y_test_energy, energy_predictions, "Energy"),
        get_metrics(Y_test_danceability, danceability_predictions, "Danceability"),
        get_metrics(Y_test_liveness, liveness_predictions, "Liveness"),
        get_metrics(Y_test_valence, valence_predictions, "Valence"),
        get_metrics(Y_test_length, length_predictions, "Length"),
        get_metrics(Y_test_acousticness, acousticness_predictions, "Acousticness"),
        get_metrics(Y_test_speechiness, speechiness_predictions, "Speechiness"),
        get_metrics(Y_test_popularity, popularity_predictions, "Popularity")
    ]
    energy_predictions = model_energy.predict(X_test_energy)
    danceability_predictions = model_danceability.predict(X_test_danceability)
    liveness_predictions = model_liveness.predict(X_test_liveness)
    valence_predictions = model_valence.predict(X_test_valence)
    length_predictions = model_length.predict(X_test_length)
    acousticness_predictions = model_acousticness.predict(X_test_acousticness)
    speechiness_predictions = model_speechiness.predict(X_test_speechiness)
    popularity_predictions = model_popularity.predict(X_test_popularity)
    
    predictions = [energy_predictions,danceability_predictions,liveness_predictions,valence_predictions,length_predictions,acousticness_predictions,speechiness_predictions,popularity_predictions]
    # Convert metrics list to JSON format
    metrics_json = json.dumps(metrics_list, indent=4)

    print(metrics_json)
    print("Predicted Energy:", energy_predictions)
    print("Predicted Danceability:", danceability_predictions)
    print("Predicted Liveness:", liveness_predictions)
    print("Predicted Valence:", valence_predictions)
    print("Predicted Length:", length_predictions)
    print("Predicted Acousticness:", acousticness_predictions)
    print("Predicted Speechiness:", speechiness_predictions)
    print("Predicted Popularity:", popularity_predictions)

# Evaluate the models and print evaluation metrics
def print_metrics(y_true, y_pred, attribute_name):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"Metrics for {attribute_name}:")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R-squared: {r2}")
    print("")

song = r"C:\Users\SOUJASH\Desktop\MusicAnalyser\musical\The Chainsmokers, Bebe Rexha - Call You Mine (Lyrics).mp3"
df2 = dfp.generate_dataset_from_file(song)
predictUsingModel(df2)
