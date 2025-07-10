"""
Script que treina os 4 modelos com as melhores configurações e testa nos dados cegos.
Avalia:
- Regressores (RandomForest, MLP): RMSE para predição de gravidade
- Classificadores (XGBoost, MLP): Precision e Recall para predição de classe
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import mean_squared_error, precision_recall_fscore_support, classification_report
import warnings
warnings.filterwarnings('ignore')

def load_training_data():
    """Carrega os dados de treinamento (4000v)."""
    # Usar dados do diretório pai para treinamento
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(parent_dir, "data_4000v", "env_vital_signals.csv")
    df = pd.read_csv(csv_path)
    return df

def load_test_data():
    """Carrega os dados cegos e os dados reais."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Dados cegos e reais estão no mesmo diretório
    cego_path = os.path.join(current_dir, "env_vital_signals_cego.csv")
    real_path = os.path.join(current_dir, "env_vital_signals.csv")
    
    df_cego = pd.read_csv(cego_path)
    df_real = pd.read_csv(real_path)
    
    return df_cego, df_real

def feature_engineering(df):
    """Aplica feature engineering."""
    df = df.copy()
    
    # Feature engineering
    df["qPA_pulso_ratio"] = df["qPA"] / (df["pulso"].replace(0, np.nan))
    df["pulso_freq_prod"] = df["pulso"] * df["freq_resp"]
    df["qPA_minus_pulso"] = df["qPA"] - df["pulso"]
    
    feature_cols = [
        "qPA", "pulso", "freq_resp",
        "qPA_pulso_ratio", "pulso_freq_prod", "qPA_minus_pulso",
    ]
    
    X = df[feature_cols].fillna(0).values
    return X

def augment_jitter(X, y, noise_level=0.03, n_copies=3):
    """Data augmentation com jitter para regressão."""
    X_aug = [X]
    y_aug = [y]
    for _ in range(n_copies):
        noise = np.random.normal(0, noise_level, X.shape)
        X_aug.append(X + noise)
        y_aug.append(y)
    return np.vstack(X_aug), np.hstack(y_aug)

def train_models():
    """Treina os 4 modelos com as melhores configurações."""
    print("Carregando dados de treinamento...")
    df_train = load_training_data()
    
    # Feature engineering
    X_train = feature_engineering(df_train)
    y_reg = df_train["grav"].values
    y_clf = df_train["classe"].values
    
    # Normalização
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    # Data augmentation para regressão
    X_jit_reg, y_jit_reg = augment_jitter(X_scaled, y_reg)
    
    # SMOTE para classificação
    le = LabelEncoder()
    y_clf_enc = le.fit_transform(y_clf)
    X_smote, y_smote = SMOTE(random_state=42).fit_resample(X_scaled, y_clf_enc)
    
    print("Treinando modelos com as melhores configurações...")
    
    # Melhores configurações fornecidas
    models = {}
    
    # 1. Random Forest Regressor
    rf_reg_config = {
        'n_estimators': 300, 
        'max_depth': 30, 
        'min_samples_split': 5, 
        'n_jobs': -1,
        'random_state': 42
    }
    models['rf_reg'] = RandomForestRegressor(**rf_reg_config)
    models['rf_reg'].fit(X_jit_reg, y_jit_reg)
    print("✓ Random Forest Regressor treinado")
    
    # 2. MLP Regressor
    mlp_reg_config = {
        'hidden_layer_sizes': (128, 64, 32), 
        'alpha': 0.0005, 
        'max_iter': 2000,
        'random_state': 42
    }
    models['mlp_reg'] = MLPRegressor(**mlp_reg_config)
    models['mlp_reg'].fit(X_jit_reg, y_jit_reg)
    print("✓ MLP Regressor treinado")
    
    # 3. XGBoost Classifier
    xgb_clf_config = {
        'n_estimators': 400, 
        'learning_rate': 0.03, 
        'max_depth': 7, 
        'subsample': 0.8, 
        'n_jobs': -1,
        'random_state': 42
    }
    models['xgb_clf'] = XGBClassifier(**xgb_clf_config)
    models['xgb_clf'].fit(X_smote, y_smote)
    print("✓ XGBoost Classifier treinado")
    
    # 4. MLP Classifier
    mlp_clf_config = {
        'hidden_layer_sizes': (128, 64), 
        'alpha': 0.0005, 
        'max_iter': 2000,
        'random_state': 42
    }
    models['mlp_clf'] = MLPClassifier(**mlp_clf_config)
    models['mlp_clf'].fit(X_smote, y_smote)
    print("✓ MLP Classifier treinado")
    
    return models, scaler, le

def evaluate_regressors(models, X_scaled, y_true):
    """Avalia os modelos de regressão usando RMSE."""
    print("\n" + "="*60)
    print("AVALIAÇÃO DOS REGRESSORES (RMSE)")
    print("="*60)
    
    for name in ['rf_reg', 'mlp_reg']:
        model = models[name]
        y_pred = model.predict(X_scaled)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        model_name = "Random Forest" if name == 'rf_reg' else "MLP (Rede Neural)"
        print(f"\n{model_name}: RMSE = {rmse:.4f}")

def evaluate_classifiers(models, le, X_scaled, y_true):
    """Avalia os modelos de classificação usando Precision e Recall."""
    print("\n" + "="*60)
    print("AVALIAÇÃO DOS CLASSIFICADORES (Precision & Recall)")
    print("="*60)
    
    # Codificar as classes reais
    y_true_encoded = le.transform(y_true)
    
    for name in ['xgb_clf', 'mlp_clf']:
        model = models[name]
        y_pred = model.predict(X_scaled)
        
        # Calcular precision e recall
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true_encoded, y_pred, average='weighted'
        )
        
        model_name = "XGBoost (Árvore)" if name == 'xgb_clf' else "MLP (Rede Neural)"
        print(f"\n{model_name}: Precision = {precision:.4f}, Recall = {recall:.4f}")

def main():
    """Função principal."""
    print("="*60)
    print("TREINAMENTO E TESTE DOS MODELOS NOS DADOS CEGOS")
    print("="*60)
    
    # Treinar modelos
    models, scaler, le = train_models()
    
    # Carregar dados de teste
    print("\nCarregando dados de teste...")
    df_cego, df_real = load_test_data()
    
    # Feature engineering nos dados cegos
    print("Aplicando feature engineering nos dados cegos...")
    X_cego = feature_engineering(df_cego)
    
    # Normalizar os dados cegos usando o scaler treinado
    X_scaled = scaler.transform(X_cego)
    
    # Valores reais (targets)
    y_real_grav = df_real['grav'].values
    y_real_classe = df_real['classe'].values
    
    print(f"\nDados carregados:")
    print(f"  - {len(df_cego)} amostras nos dados cegos")
    print(f"  - {len(df_real)} amostras nos dados reais")
    print(f"  - {X_scaled.shape[1]} features")
    
    # Avaliar regressores
    evaluate_regressors(models, X_scaled, y_real_grav)
    
    # Avaliar classificadores
    evaluate_classifiers(models, le, X_scaled, y_real_classe)
    
    print("\n" + "="*60)
    print("AVALIAÇÃO CONCLUÍDA!")
    print("="*60)

if __name__ == "__main__":
    main()
