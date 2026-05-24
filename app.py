from flask import Flask, request, jsonify, render_template
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import numpy as np
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
    smiles = data.get('smiles', '')

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return jsonify({'error': 'Invalid SMILES string'})

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
    app.run(debug=True)