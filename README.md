# Drug-Target Interaction Predictor

A machine learning web app that predicts whether a compound is active against Plasmodium falciparum (Malaria).

## Live Demo
https://drug-target-interaction-predictor.onrender.com/

## About
- Algorithm: Random Forest
- Features: Morgan Fingerprints (2048 bits)
- Accuracy: 86%
- ROC-AUC: 0.927
- Training data: 27,950 compounds from ChEMBL

## Tech Stack
- Python, Flask, RDKit, Scikit-learn
- HTML, CSS, JavaScript

## How to run locally
```bash
pip install -r requirements.txt
python app.py
```