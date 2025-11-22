import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import warnings

# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore")

# ==========================================
# 1. Data Simulation 
# ==========================================
def generate_mock_dataset(n_samples=200):
    print("[1/5] Generating synthetic Multi-Omics data...")
    
    # Valid SMILES strings for simulation
    smiles_db = [
        "CC(C)CCCC(C)C1CCC2C3CC=C4CC(O)CCC4(C)C3CCC12C", # Cholesterol (Triterpenoid scaffold)
        "OCC1OC(O)C(O)C(O)C1O", # Glucose
        "COc1cc(C=CC(=O)O)cc(OC)c1O", # Sinapic acid
        "CC(=O)OC1CC2CCC3C(C)(C(=O)O)CCC3(C)C2(C)CC1=O", # Mock Ganoderic Acid core
        "CCCCCCCCCC(=O)O", # Fatty Acid
        "C1=CC=C(C=C1)O" # Phenol
    ]
    
    data = []
    for i in range(n_samples):
        # Pick a random structure
        smiles = np.random.choice(smiles_db)
        
        # Logic: Triterpenoids ("Ganoderic" analogs) are bioactive
        is_bioactive = 1 if len(smiles) > 30 else 0 
        
        # Add noise to make it realistic (approx. 10% label noise)
        if np.random.random() > 0.9:
            is_bioactive = 1 - is_bioactive

        row = {
            "UFID": f"FUNG_{i:04d}",
            "SMILES": smiles,
            # Transcriptomics: Bioactives usually have higher expression
            "Gene_Expression": np.random.normal(2.5, 0.5) if is_bioactive else np.random.normal(0.5, 0.5),
            # Genomics: Higher BGC confidence
            "BGC_Score": np.random.uniform(0.7, 1.0) if is_bioactive else np.random.uniform(0.0, 0.5),
            # Pathway Completeness
            "Pathway_Completeness": np.random.uniform(0.8, 1.0) if is_bioactive else np.random.uniform(0.1, 0.6),
            "Bioactivity_Label": is_bioactive
        }
        data.append(row)
        
    return pd.DataFrame(data)

# ==========================================
# 2. Feature Engineering (RDKit)
# ==========================================
class SysFungiNetPreprocessor:
    def compute_fingerprints(self, df):
        print("[2/5] Computing Cheminformatic Descriptors...")
        fps = []
        mol_props = []
        
        for smiles in df['SMILES']:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                # Reduced bits to 256 for speed/stability in demo
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=256)
                fps.append(list(fp))
                mol_props.append([Descriptors.MolWt(mol), Descriptors.MolLogP(mol)])
            else:
                # Fallback for invalid SMILES
                fps.append([0]*256)
                mol_props.append([0, 0])
                
        # Create DataFrames
        fp_df = pd.DataFrame(fps, columns=[f'FP_{i}' for i in range(256)])
        prop_df = pd.DataFrame(mol_props, columns=['MolWt', 'LogP'])
        
        # Concatenate all features
        # Note: We exclude string columns (UFID, SMILES) and Target (Bioactivity_Label)
        omics_cols = df[['Gene_Expression', 'BGC_Score', 'Pathway_Completeness']].reset_index(drop=True)
        
        X = pd.concat([omics_cols, prop_df, fp_df], axis=1)
        return X

# ==========================================
# 3. Model & Explainability
# ==========================================
class SysFungiNetModel:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
        
    def train(self, X, y):
        print("[3/5] Training Ensemble Model (XGBoost)...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(self.X_train, self.y_train)
        
        preds = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, preds)
        print(f"    - Model Accuracy: {acc:.2f}")
        return self.X_test

    def explain(self, X_sample):
        print("[4/5] Generating SHAP Explanations...")
        
        # Use TreeExplainer
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_sample)
        
        # ERROR FIX: Handle SHAP return type for Classification
        # Some versions return a list [class0_values, class1_values], others return a single array.
        if isinstance(shap_values, list):
            vals = np.abs(shap_values[1]) # Take positive class
        else:
            vals = np.abs(shap_values)

        # Calculate mean importance
        mean_shap = vals.mean(axis=0)
        
        # Get top features
        top_indices = np.argsort(mean_shap)[-5:][::-1]
        
        print("\n    === Top Predictive Features (SHAP Analysis) ===")
        print("    (Higher value = Stronger driver of Nutraceutical Potential)")
        feature_names = X_sample.columns.tolist()
        for idx in top_indices:
            print(f"    - {feature_names[idx]}: {mean_shap[idx]:.4f}")

# ==========================================
# 4. Pipeline Execution
# ==========================================
def run_pipeline():
    # 1. Generate Data
    df = generate_mock_dataset()
    y = df['Bioactivity_Label']
    
    # 2. Preprocessing
    pp = SysFungiNetPreprocessor()
    X = pp.compute_fingerprints(df)
    
    # 3. Training
    pipeline = SysFungiNetModel()
    X_test = pipeline.train(X, y)
    
    # 4. Explanation (on first 10 test samples)
    pipeline.explain(X_test.iloc[:10])
    
    # 5. Prioritization Output
    print("\n[5/5] Prioritizing Candidates (Simulation)...")
    
    # Calculate scores for the whole dataset
    nps_scores = pipeline.model.predict_proba(X)[:, 1] # Probability of Class 1
    df['NPS_Score'] = nps_scores
    
    # Mock Docking Energy
    df['Docking_Energy'] = np.random.uniform(-10.0, -5.0, size=len(df))
    
    # Show Top 5 Candidates
    results = df.sort_values(by='NPS_Score', ascending=False).head(5)
    print("\n" + "="*60)
    print("FINAL OUTPUT: Prioritized Nutraceutical Candidates")
    print("="*60)
    print(results[['UFID', 'SMILES', 'NPS_Score', 'Docking_Energy']].to_string(index=False))

if __name__ == "__main__":
    run_pipeline()