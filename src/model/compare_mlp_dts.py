"""
Treinamento e avaliação de 4 modelos (RFReg, MLPReg, XGBClf, MLPClf)
com **3 configurações** cada. O script imprime as métricas de cada
configuração usando validação cruzada (5‑fold) e salva o scaler e os
encoders para uso posterior.

• Regressão → métrica: RMSE (quanto menor, melhor)
• Classificação → métrica: Acurácia (quanto maior, melhor)
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Caminho dos dados
ROOT = os.path.dirname(__file__)
csv_path = os.path.join(ROOT, "data_4000v", "env_vital_signals.csv")
df = pd.read_csv(csv_path)

# Feature engineering ------------------------------------------------------
df["qPA_pulso_ratio"] = df["qPA"] / (df["pulso"].replace(0, np.nan))
df["pulso_freq_prod"] = df["pulso"] * df["freq_resp"]
df["qPA_minus_pulso"] = df["qPA"] - df["pulso"]

feature_cols = [
    "qPA", "pulso", "freq_resp",
    "qPA_pulso_ratio", "pulso_freq_prod", "qPA_minus_pulso",
]

X = df[feature_cols].fillna(0).values
y_reg = df["grav"].values
y_clf = df["classe"].values

# Normalização -------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------------------------------------------------
# Data augmentation: jitter para regressão, SMOTE para classificação -------
# -------------------------------------------------------------------------
def augment_jitter(X, y, noise_level=0.03, n_copies=3):
    X_aug = [X]
    y_aug = [y]
    for _ in range(n_copies):
        noise = np.random.normal(0, noise_level, X.shape)
        X_aug.append(X + noise)
        y_aug.append(y)
    return np.vstack(X_aug), np.hstack(y_aug)

# Jitter para regressão
X_jit_reg, y_jit_reg = augment_jitter(X_scaled, y_reg)

# SMOTE para classificação
le = LabelEncoder()
y_clf_enc = le.fit_transform(y_clf)
X_smote, y_smote = SMOTE(random_state=42).fit_resample(X_scaled, y_clf_enc)

# -------------------------------------------------------------------------
# Função genérica de avaliação (k‑fold CV) ---------------------------------
# -------------------------------------------------------------------------

def avaliar_configs(model_class, cfgs, X, y, scoring, nome_modelo):
    """Avalia uma lista de configurações para um dado modelo.

    Retorna lista de tuplas: (cfg, média_score).
    """
    resultados = []
    print(f"\n=== {nome_modelo} ===")
    for idx, cfg in enumerate(cfgs, 1):
        cfg_extra = cfg.copy()
        if "random_state" not in cfg_extra:
            cfg_extra["random_state"] = 42
        try:
            modelo = model_class(**cfg_extra)
        except TypeError:
            cfg_extra.pop("n_jobs", None)
            modelo = model_class(**cfg_extra)

        scores = cross_val_score(
            modelo,
            X,
            y,
            scoring=scoring,
            cv=5,
            n_jobs=-1,
        )
        media = (-scores.mean()) if scoring.startswith("neg_") else scores.mean()
        print(f"Config {idx}: {cfg} → {scoring.replace('neg_', '').upper()} = {media:.4f}")
        resultados.append((cfg, media))
    if scoring.startswith("neg_"):
        melhor = max(resultados, key=lambda x: x[1])  # menos negativo = melhor
    else:
        melhor = max(resultados, key=lambda x: x[1])
    return resultados, melhor

# -------------------------------------------------------------------------
# Configurações para cada modelo ------------------------------------------
# -------------------------------------------------------------------------

rf_reg_cfgs = [
    {"n_estimators": 200, "max_depth": 20, "min_samples_split": 2, "n_jobs": -1},
    {"n_estimators": 300, "max_depth": 30, "min_samples_split": 5, "n_jobs": -1},
    {"n_estimators": 400, "max_depth": None, "min_samples_split": 2, "n_jobs": -1},
]

mlp_reg_cfgs = [
    {"hidden_layer_sizes": (64, 32), "alpha": 1e-4, "max_iter": 1000},
    {"hidden_layer_sizes": (128, 64, 32), "alpha": 5e-4, "max_iter": 2000},
    {"hidden_layer_sizes": (256, 128, 64), "alpha": 1e-3, "max_iter": 2500},
]

xgb_clf_cfgs = [
    {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8, "n_jobs": -1},
    {"n_estimators": 200, "learning_rate": 0.10, "max_depth": 5, "subsample": 0.9, "n_jobs": -1},
    {"n_estimators": 400, "learning_rate": 0.03, "max_depth": 7, "subsample": 0.8, "n_jobs": -1},
]

mlp_clf_cfgs = [
    {"hidden_layer_sizes": (64, 32), "alpha": 1e-4, "max_iter": 1000},
    {"hidden_layer_sizes": (128, 64), "alpha": 5e-4, "max_iter": 2000},
    {"hidden_layer_sizes": (256, 128, 64), "alpha": 1e-3, "max_iter": 2500},
]

# -------------------------------------------------------------------------
# Avaliação ----------------------------------------------------------------
# -------------------------------------------------------------------------

# Regressão (com jitter)
rf_results, rf_best = avaliar_configs(RandomForestRegressor, rf_reg_cfgs, X_jit_reg, y_jit_reg, scoring="neg_root_mean_squared_error", nome_modelo="RandomForestRegressor")
mlp_reg_results, mlp_reg_best = avaliar_configs(MLPRegressor, mlp_reg_cfgs, X_jit_reg, y_jit_reg, scoring="neg_root_mean_squared_error", nome_modelo="MLPRegressor")

# Classificação (com SMOTE)
xgb_results, xgb_best = avaliar_configs(XGBClassifier, xgb_clf_cfgs, X_smote, y_smote, scoring="accuracy", nome_modelo="XGBClassifier")
mlp_clf_results, mlp_clf_best = avaliar_configs(MLPClassifier, mlp_clf_cfgs, X_smote, y_smote, scoring="accuracy", nome_modelo="MLPClassifier")

print("\n✅ Avaliação concluída! Veja as métricas acima para cada configuração.")

# -------------------------------------------------------------------------
# Exibe as melhores configurações -----------------------------------------
# -------------------------------------------------------------------------
print("\n=== MELHORES CONFIGURAÇÕES ===")
print(f"RandomForestRegressor: {rf_best[0]} → RMSE = {abs(rf_best[1]):.4f}")
print(f"MLPRegressor:         {mlp_reg_best[0]} → RMSE = {abs(mlp_reg_best[1]):.4f}")
print(f"XGBClassifier:        {xgb_best[0]} → ACC  = {xgb_best[1]:.4f}")
print(f"MLPClassifier:        {mlp_clf_best[0]} → ACC  = {mlp_clf_best[1]:.4f}")

# -------------------------------------------------------------------------
# Métricas de precisão e recall para classificação ------------------------
# -------------------------------------------------------------------------
from sklearn.model_selection import cross_val_score

# XGBClassifier
prec_xgb = cross_val_score(XGBClassifier(**xgb_best[0]), X_smote, y_smote, scoring="precision_macro", cv=5, n_jobs=-1)
rec_xgb  = cross_val_score(XGBClassifier(**xgb_best[0]), X_smote, y_smote, scoring="recall_macro", cv=5, n_jobs=-1)

# MLPClassifier
prec_mlp = cross_val_score(MLPClassifier(**mlp_clf_best[0]), X_smote, y_smote, scoring="precision_macro", cv=5, n_jobs=-1)
rec_mlp  = cross_val_score(MLPClassifier(**mlp_clf_best[0]), X_smote, y_smote, scoring="recall_macro", cv=5, n_jobs=-1)

print("\n=== MÉTRICAS DE PRECISÃO E RECALL (macro, média dos folds) ===")
print(f"XGBClassifier (Árvore): Precision = {prec_xgb.mean():.4f} | Recall = {rec_xgb.mean():.4f}")
print(f"MLPClassifier (Rede NN): Precision = {prec_mlp.mean():.4f} | Recall = {rec_mlp.mean():.4f}")