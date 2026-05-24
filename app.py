import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

app = Flask(__name__)

# ── TRAIN OR LOAD MODEL ───────────────────────────────────────
def get_model():
    model_path = 'rf_model.pkl'

    if os.path.exists(model_path):
        print("Loading existing model...")
        with open(model_path, 'rb') as f:
            model, generator = pickle.load(f)
        return model, generator

    print("Training model from ChEMBL...")
    from chembl_webresource_client.new_client import new_client

    activity = new_client.activity
    data = activity.filter(
        target_chembl_id='CHEMBL364',
        standard_type='IC50'
    )
    df = pd.DataFrame.from_records(data)

    df = df[['molecule_chembl_id', 'canonical_smiles', 'standard_value']]
    df = df.dropna()
    df['standard_value'] = pd.to_numeric(df['standard_value'], errors='coerce')
    df = df.dropna()
    df = df.drop_duplicates(subset='molecule_chembl_id')
    df['label'] = df['standard_value'].apply(lambda x: 1 if x < 1000 else 0)

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def to_fp(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return list(generator.GetFingerprint(mol))

    df['fp'] = df['canonical_smiles'].apply(to_fp)
    df = df.dropna(subset=['fp'])

    X = np.array(df['fp'].tolist())
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save for next startup
    with open(model_path, 'wb') as f:
        pickle.dump((model, generator), f)

    print("Model trained and saved!")
    return model, generator

model, generator = get_model()

# ── ROUTES ────────────────────────────────────────────────────
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
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)