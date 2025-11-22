import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ==============================================================================
# PHASE 1: DATA GENERATION (Simulating the 312 Metabolites from the Abstract)
# ==============================================================================
def generate_journal_dataset():
    print("\n[PHASE 1] Generating SysFungiNet Multi-Omics Dataset (N=312)...")
    
    num_samples = 312
    species_list = ["Ganoderma lucidum", "Craterellus cornucopioides"]
    
    # --- Molecular Structures (SMILES) ---
    # Triterpenoids (Ganoderma-like)
    triterpenoids = [
        "CC(=O)OC1CC2CCC3C(C)(C(=O)O)CCC3(C)C2(C)CC1=O", # Ganoderic acid core
        "CC(C)CCCC(C)C1CCC2C3CC=C4CC(O)CCC4(C)C3CCC12C", # Lanostane scaffold
    ]
    # Phenolics (Craterellus-like)
    phenolics = [
        "COc1cc(C=CC(=O)O)cc(OC)c1O", # Sinapic Acid
        "Oc1ccc(C=CC(=O)O)cc1", # Coumaric Acid
    ]
    # Inactive/Common background molecules
    common = [
        "OCC1OC(O)C(O)C(O)C1O", "CCCCCCCCCC(=O)O", "NCC(=O)O", "CC(C)C(N)C(=O)O"
    ]

    data = []
    
    for i in range(num_samples):
        ufid = f"UFID_{i:04d}"
        species = np.random.choice(species_list, p=[0.6, 0.4])
        
        # Determine Class and Structure
        is_bioactive = False
        chem_name = "Unknown_Metabolite"
        
        if species == "Ganoderma lucidum":
            if np.random.random() > 0.75: 
                smiles = np.random.choice(triterpenoids)
                chem_class = "Triterpenoid"
                is_bioactive = True
            else:
                smiles = np.random.choice(common)
                chem_class = "Primary Metabolite"
        else:
            if np.random.random() > 0.75:
                smiles = np.random.choice(phenolics)
                chem_class = "Phenolic"
                is_bioactive = True
            else:
                smiles = np.random.choice(common)
                chem_class = "Primary Metabolite"

        # --- Inject the "Cornucopiolide" Discovery (Rank 2 in Table 4) ---
        if i == 50: 
            species = "Craterellus cornucopioides"
            chem_name = "Cornucopiolide (Putative)"
            smiles = "CC1(C)CCCC2(C)C1CCC(=C)C2CCC(=O)OC" # Unique Terpenoid
            chem_class = "Terpenoid"
            is_bioactive = True

        # --- Simulate Omics Data ---
        # Bioactives have correlated High Expression + High Pathway Completeness
        if is_bioactive:
            log2fc = np.random.normal(2.5, 0.6)  # High Expression
            bgc_conf = np.random.uniform(0.7, 1.0) # High Genomic Confidence
            pci = np.random.uniform(0.75, 0.95)   # High Pathway Completeness
            docking = np.random.uniform(-10.5, -8.0) # Strong Binding
        else:
            log2fc = np.random.normal(0.5, 0.8)
            bgc_conf = np.random.uniform(0.0, 0.5)
            pci = np.random.uniform(0.1, 0.6)
            docking = np.random.uniform(-6.0, -3.0)

        # Add Noise (Biological Variation)
        if np.random.random() > 0.95: log2fc += 1.5 # Random noise

        data.append({
            "UFID": ufid,
            "Species": species,
            "Chemical_Name": chem_name,
            "SMILES": smiles,
            "Log2FC_Expression": round(log2fc, 3),        # Transcriptomics
            "BGC_Confidence_Score": round(bgc_conf, 3),   # Genomics
            "Pathway_Completeness_Index": round(pci, 3),  # Systems Bio
            "Is_Nutraceutical": 1 if is_bioactive else 0, # Label
            "Docking_Energy_Ref": round(docking, 2)       # For validation comparison
        })

    df = pd.DataFrame(data)
    df.to_csv("sysfunginet_dataset_Q1.csv", index=False)
    print(" -> Data generated and saved to 'sysfunginet_dataset_Q1.csv'.")
    return df

# ==============================================================================
# PHASE 2: PREPROCESSING & FEATURE ENGINEERING (Section 3.6.1)
# ==============================================================================
class SysFungiNetPreprocessor:
    def process(self, df):
        print("\n[PHASE 2] Feature Engineering (RDKit + Omics Integration)...")
        fps = []
        props = []
        
        for smiles in df['SMILES']:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                # ECFP4 Fingerprints (Structure)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=256)
                fps.append(list(fp))
                # Physicochemical Props
                props.append([Descriptors.MolWt(mol), Descriptors.MolLogP(mol)])
            else:
                fps.append([0]*256)
                props.append([0, 0])
                
        # Feature DataFrames
        fp_df = pd.DataFrame(fps, columns=[f'FP_{i}' for i in range(256)])
        prop_df = pd.DataFrame(props, columns=['MolWt', 'LogP'])
        
        # Select Omics Columns
        omics_df = df[['Log2FC_Expression', 'BGC_Confidence_Score', 'Pathway_Completeness_Index']].reset_index(drop=True)
        
        # Integration: Combine Structure + Omics
        X = pd.concat([omics_df, prop_df, fp_df], axis=1)
        return X

# ==============================================================================
# PHASE 3: MODEL TRAINING & EXPLAINABILITY (Section 3.6.2 & 3.6.3)
# ==============================================================================
class SysFungiNetModel:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
        
    def train(self, X, y):
        print("\n[PHASE 3] Training XGBoost Ensemble Model...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(self.X_train, self.y_train)
        
        # Benchmark Metrics
        preds = self.model.predict(self.X_test)
        f1 = f1_score(self.y_test, preds)
        print(f" -> Model Performance on Test Set:")
        print(f"    - Accuracy: {accuracy_score(self.y_test, preds):.2f}")
        print(f"    - F1-Score: {f1:.2f} (Target: >0.90)")
        return self.X_test

    def explain(self, X_sample):
        print("\n[PHASE 4] Explainable AI (SHAP Analysis)...")
        
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_sample)
        
        # Handle SHAP output format variations
        vals = np.abs(shap_values[1]) if isinstance(shap_values, list) else np.abs(shap_values)
        
        # Calculate Mean Absolute SHAP values
        mean_shap = vals.mean(axis=0)
        top_indices = np.argsort(mean_shap)[-5:][::-1]
        
        print(" -> Top 5 Drivers of Nutraceutical Potential:")
        feature_names = X_sample.columns.tolist()
        for idx in top_indices:
            print(f"    {feature_names[idx]}: Impact {mean_shap[idx]:.4f}")

# ==============================================================================
# MAIN EXECUTION FLOW
# ==============================================================================
def run_sysfunginet():
    # 1. Load Data
    if not os.path.exists("sysfunginet_dataset_Q1.csv"):
        df = generate_journal_dataset()
    else:
        print("Loading existing dataset...")
        df = pd.read_csv("sysfunginet_dataset_Q1.csv")

    y = df['Is_Nutraceutical']
    
    # 2. Preprocessing
    pp = SysFungiNetPreprocessor()
    X = pp.process(df)
    
    # 3. Train Model
    pipeline = SysFungiNetModel()
    X_test = pipeline.train(X, y)
    
    # 4. Explainability (Analyze first 20 samples)
    pipeline.explain(X_test.iloc[:20])
    
    # 5. Prioritization & Output
    print("\n[PHASE 5] Prioritizing Candidates (Table 4 Simulation)...")
    
    # Predict "Nutraceutical Potential Score" (NPS)
    df['NPS_Score'] = pipeline.model.predict_proba(X)[:, 1]
    
    # Filter candidates (High Probability)
    candidates = df[df['NPS_Score'] > 0.85].copy()
    
    # Sort by NPS and Docking Energy
    top_candidates = candidates.sort_values(by=['NPS_Score'], ascending=False).head(5)
    
    print("\n" + "="*80)
    print(f"{'UFID':<10} | {'Chemical Name':<25} | {'Species':<20} | {'NPS':<5} | {'PCI':<5} | {'Docking (kcal/mol)':<15}")
    print("="*80)
    
    for _, row in top_candidates.iterrows():
        print(f"{row['UFID']:<10} | {row['Chemical_Name']:<25} | {row['Species'].split()[0]:<20} | {row['NPS_Score']:.2f}  | {row['Pathway_Completeness_Index']:.2f}  | {row['Docking_Energy_Ref']:.2f}")
    print("="*80)
    print("Note: 'Cornucopiolide' should appear in top ranks with High NPS and Strong Docking (<-9.0).")

if __name__ == "__main__":
    run_sysfunginet()