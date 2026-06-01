import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import pickle

app = Flask(__name__)

# Load model
with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    smiles = data.get('smiles', '').strip()

    if not smiles:
        return jsonify({'error': 'Please enter a SMILES string'})

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return jsonify({'error': 'Invalid SMILES string. Please check and try again'})

    fp = generator.GetFingerprint(mol)
    fp_array = np.array(fp).reshape(1, -1)

    pred = model.predict(fp_array)[0]
    prob = model.predict_proba(fp_array)[0]

    return jsonify({
        'prediction': 'ACTIVE' if pred == 1 else 'INACTIVE',
        'active_prob': round(float(prob[1]) * 100, 1),
        'inactive_prob': round(float(prob[0]) * 100, 1)
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)