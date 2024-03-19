from flask import Flask, request, jsonify
import model

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get file path from request
    song_file = request.files['song']

    # Generate dataset from file
    df2 = model.dfp.generate_dataset_from_file(song_file)

    # Predict using model
    predictions = model.predictUsingModel(df2)

    # Return predictions as JSON response
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
