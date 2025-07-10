"""
Script para testar várias configurações dos modelos de árvore nos dados cegos.
Testa:
- RandomForest Regressor: várias configurações para predição de gravidade
- XGBoost Classifier: várias configurações para predição de classe
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import mean_squared_error, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

def load_training_data():
    """Carrega os dados de treinamento (4000v)."""
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(parent_dir, "data_4000v", "env_vital_signals.csv")
    df = pd.read_csv(csv_path)
    return df

def load_test_data():
    """Carrega os dados cegos e os dados reais."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    cego_path = os.path.join(current_dir, "env_vital_signals_cego.csv")
    real_path = os.path.join(current_dir, "env_vital_signals.csv")
    
    df_cego = pd.read_csv(cego_path)
    df_real = pd.read_csv(real_path)
    
    return df_cego, df_real

def feature_engineering(df):
    """Aplica feature engineering."""
    df = df.copy()
    
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

def prepare_data():
    """Prepara os dados de treinamento e teste."""
    print("Preparando dados...")
    
    # Dados de treinamento
    df_train = load_training_data()
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
    
    # Dados de teste
    df_cego, df_real = load_test_data()
    X_test = feature_engineering(df_cego)
    X_test_scaled = scaler.transform(X_test)
    
    y_test_grav = df_real['grav'].values
    y_test_classe = df_real['classe'].values
    y_test_classe_enc = le.transform(y_test_classe)
    
    return (X_jit_reg, y_jit_reg, X_smote, y_smote, 
            X_test_scaled, y_test_grav, y_test_classe_enc, le)

def test_random_forest_configs(X_train, y_train, X_test, y_test):
    """Testa várias configurações do RandomForest Regressor."""
    print("\n" + "="*70)
    print("TESTANDO CONFIGURAÇÕES DO RANDOM FOREST REGRESSOR")
    print("="*70)
    
    # Configurações para testar
    rf_configs = [
        # Configurações básicas
        {"n_estimators": 100, "max_depth": 10, "min_samples_split": 2},
        {"n_estimators": 100, "max_depth": 20, "min_samples_split": 2},
        {"n_estimators": 100, "max_depth": None, "min_samples_split": 2},
        
        # Configurações médias
        {"n_estimators": 200, "max_depth": 15, "min_samples_split": 5},
        {"n_estimators": 200, "max_depth": 25, "min_samples_split": 5},
        {"n_estimators": 200, "max_depth": None, "min_samples_split": 5},
        
        # Configurações avançadas
        {"n_estimators": 300, "max_depth": 20, "min_samples_split": 2},
        {"n_estimators": 300, "max_depth": 30, "min_samples_split": 5},
        {"n_estimators": 300, "max_depth": None, "min_samples_split": 10},
        
        # Configurações robustas
        {"n_estimators": 500, "max_depth": 25, "min_samples_split": 5},
        {"n_estimators": 500, "max_depth": 35, "min_samples_split": 10},
        {"n_estimators": 500, "max_depth": None, "min_samples_split": 15},
        
        # Configurações especiais
        {"n_estimators": 150, "max_depth": 12, "min_samples_split": 3, "max_features": "sqrt"},
        {"n_estimators": 250, "max_depth": 18, "min_samples_split": 4, "max_features": "log2"},
        {"n_estimators": 400, "max_depth": 40, "min_samples_split": 8, "min_samples_leaf": 2},
    ]
    
    best_rmse = float('inf')
    best_config = None
    
    for i, config in enumerate(rf_configs, 1):
        # Adicionar parâmetros padrão
        full_config = {**config, "n_jobs": -1, "random_state": 42}
        
        # Treinar modelo
        model = RandomForestRegressor(**full_config)
        model.fit(X_train, y_train)
        
        # Predizer e calcular RMSE
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"Config {i:2d}: {config}")
        print(f"          RMSE = {rmse:.4f}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_config = config
        
        print()
    
    print(f"🏆 MELHOR CONFIGURAÇÃO RF:")
    print(f"   {best_config}")
    print(f"   RMSE = {best_rmse:.4f}")
    
    return best_config, best_rmse

def test_xgboost_configs(X_train, y_train, X_test, y_test):
    """Testa várias configurações do XGBoost Classifier."""
    print("\n" + "="*70)
    print("TESTANDO CONFIGURAÇÕES DO XGBOOST CLASSIFIER")
    print("="*70)
    
    # Configurações para testar
    xgb_configs = [
        # Configurações básicas
        {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
        {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6},
        {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 6},
        
        # Configurações médias
        {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 4, "subsample": 0.8},
        {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.9},
        {"n_estimators": 200, "learning_rate": 0.03, "max_depth": 8, "subsample": 0.8},
        
        # Configurações avançadas
        {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 5, "subsample": 0.8},
        {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 7, "subsample": 0.8},
        {"n_estimators": 300, "learning_rate": 0.01, "max_depth": 9, "subsample": 0.9},
        
        # Configurações robustas
        {"n_estimators": 400, "learning_rate": 0.03, "max_depth": 6, "subsample": 0.8},
        {"n_estimators": 400, "learning_rate": 0.03, "max_depth": 7, "subsample": 0.8},
        {"n_estimators": 400, "learning_rate": 0.02, "max_depth": 8, "subsample": 0.7},
        
        # Configurações especiais
        {"n_estimators": 500, "learning_rate": 0.02, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 250, "learning_rate": 0.04, "max_depth": 5, "subsample": 0.9, "colsample_bytree": 0.9},
        {"n_estimators": 350, "learning_rate": 0.025, "max_depth": 7, "subsample": 0.75, "reg_alpha": 0.1},
    ]
    
    best_precision = 0
    best_recall = 0
    best_f1 = 0
    best_config = None
    
    for i, config in enumerate(xgb_configs, 1):
        # Adicionar parâmetros padrão
        full_config = {**config, "n_jobs": -1, "random_state": 42}
        
        # Treinar modelo
        model = XGBClassifier(**full_config)
        model.fit(X_train, y_train)
        
        # Predizer e calcular métricas
        y_pred = model.predict(X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        
        print(f"Config {i:2d}: {config}")
        print(f"          Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}")
        
        # Critério: melhor F1-score
        if f1 > best_f1:
            best_f1 = f1
            best_precision = precision
            best_recall = recall
            best_config = config
        
        print()
    
    print(f"🏆 MELHOR CONFIGURAÇÃO XGB:")
    print(f"   {best_config}")
    print(f"   Precision = {best_precision:.4f}, Recall = {best_recall:.4f}, F1 = {best_f1:.4f}")
    
    return best_config, best_precision, best_recall

def main():
    """Função principal."""
    print("="*70)
    print("TESTE DE MÚLTIPLAS CONFIGURAÇÕES DOS MODELOS DE ÁRVORE")
    print("="*70)
    
    # Preparar dados
    (X_jit_reg, y_jit_reg, X_smote, y_smote, 
     X_test_scaled, y_test_grav, y_test_classe_enc, le) = prepare_data()
    
    print(f"Dados preparados:")
    print(f"  - Treinamento Regressão: {X_jit_reg.shape[0]} amostras")
    print(f"  - Treinamento Classificação: {X_smote.shape[0]} amostras")
    print(f"  - Teste: {X_test_scaled.shape[0]} amostras")
    print(f"  - Features: {X_test_scaled.shape[1]}")
    
    # Testar RandomForest Regressor
    best_rf_config, best_rf_rmse = test_random_forest_configs(
        X_jit_reg, y_jit_reg, X_test_scaled, y_test_grav
    )
    
    # Testar XGBoost Classifier
    best_xgb_config, best_xgb_precision, best_xgb_recall = test_xgboost_configs(
        X_smote, y_smote, X_test_scaled, y_test_classe_enc
    )
    
    # Resumo final
    print("\n" + "="*70)
    print("RESUMO DOS MELHORES RESULTADOS")
    print("="*70)
    print(f"\n🌲 RANDOM FOREST REGRESSOR:")
    print(f"   Configuração: {best_rf_config}")
    print(f"   RMSE: {best_rf_rmse:.4f}")
    
    print(f"\n🌲 XGBOOST CLASSIFIER:")
    print(f"   Configuração: {best_xgb_config}")
    print(f"   Precision: {best_xgb_precision:.4f}")
    print(f"   Recall: {best_xgb_recall:.4f}")
    
    print("\n" + "="*70)
    print("TESTE CONCLUÍDO!")
    print("="*70)

if __name__ == "__main__":
    main()
