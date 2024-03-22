from flask import Flask, request, jsonify
import DataFeaturing as dfp
from model import*
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def read_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        try:
            # Process the file here
            
            df2 = dfp.generate_dataset_from_file(file)
            data = predictUsingModel(df2)
            return jsonify({'message_1': file.filename,"Data":data})
        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
