import pickle
import numpy as np
import streamlit as st
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="Drug Activity Predictor",
    page_icon="🧬",
    layout="centered"
)

# ── LOAD MODEL ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()
generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# ── SESSION STATE ─────────────────────────────────────────────
if 'smiles_input' not in st.session_state:
    st.session_state.smiles_input = ""

# ── HEADER ────────────────────────────────────────────────────
st.title("🧬 Drug-Target Interaction Predictor")
st.markdown("Predicts whether a compound is **active** against *Plasmodium falciparum* (Malaria)")
st.markdown("---")

# ── INPUT ─────────────────────────────────────────────────────
smiles_input = st.text_input(
    "Enter a SMILES string:",
    placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O",
    value=st.session_state.smiles_input
)

# ── EXAMPLE BUTTONS ───────────────────────────────────────────
st.markdown("**Try an example:**")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Aspirin"):
        st.session_state.smiles_input = "CC(=O)Oc1ccccc1C(=O)O"
        st.rerun()

with col2:
    if st.button("Chloroquine"):
        st.session_state.smiles_input = "CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12"
        st.rerun()

with col3:
    if st.button("Quinine"):
        st.session_state.smiles_input = "COc1ccc2nccc([C@@H](O)[C@H]3C[C@@H]4CC[N@]3C[C@@H]4C=C)c2c1"
        st.rerun()

# ── PREDICTION ────────────────────────────────────────────────
st.markdown("---")
if st.button("🔍 Predict", type="primary"):
    if smiles_input:
        mol = Chem.MolFromSmiles(smiles_input)
        if mol:
            fp = generator.GetFingerprint(mol)
            fp_array = np.array(fp).reshape(1, -1)

            pred = model.predict(fp_array)[0]
            prob = model.predict_proba(fp_array)[0]

            st.markdown("### Result")

            if pred == 1:
                st.success("✅ ACTIVE — This compound is likely active against malaria")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Active Probability", f"{prob[1]*100:.1f}%")
                with col2:
                    st.metric("Inactive Probability", f"{prob[0]*100:.1f}%")
            else:
                st.error("❌ INACTIVE — This compound is unlikely to be active")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Active Probability", f"{prob[1]*100:.1f}%")
                with col2:
                    st.metric("Inactive Probability", f"{prob[0]*100:.1f}%")

            # Molecule details
            st.markdown("**Molecule Details:**")
            st.code(smiles_input, language=None)

        else:
            st.warning("⚠️ Invalid SMILES string. Please check and try again.")
    else:
        st.warning("⚠️ Please enter a SMILES string or click an example above.")

# ── FOOTER ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "Built with **RDKit** + **Scikit-learn** + **Streamlit** | Data from **ChEMBL**"
)