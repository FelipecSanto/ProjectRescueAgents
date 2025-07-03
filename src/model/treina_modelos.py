#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Treinamento e avaliação de modelos para sinais vitais
Estratégias implementadas
1. RandomizedSearchCV para ajuste fino de Random Forest
2. Novos modelos: HistGradientBoosting e (opcional) XGBoost
3. Ensembles com VotingClassifier / VotingRegressor
4. Feature-engineering simples
5. Visualização dos principais erros (confusion-matrix)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# 1. LEITURA DOS DADOS + FEATURES
# ---------------------------------------------------------------------
ROOT = os.path.dirname(__file__)
csv_path = os.path.join(ROOT, "data_4000v", "env_vital_signals.csv")
df = pd.read_csv(csv_path)

# ---------------- Feature‑engineering --------------------------------
df["qPA_pulso_ratio"] = df["qPA"] / (df["pulso"].replace(0, np.nan))
df["pulso_freq_prod"] = df["pulso"] * df["freq_resp"]
df["qPA_minus_pulso"] = df["qPA"] - df["pulso"]

feature_cols = [
    "qPA", "pulso", "freq_resp",
    "qPA_pulso_ratio", "pulso_freq_prod", "qPA_minus_pulso"
]

X = df[feature_cols].fillna(0).values
y_reg = df["grav"].values
y_clf = df["classe"].values

# ---------------------------------------------------------------------
# 2. NORMALIZAÇÃO
# ---------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------------------
# 3. DATA AUGMENTATION
# ---------------------------------------------------------------------
from sklearn.mixture import GaussianMixture
from imblearn.over_sampling import SMOTE

def augment_jitter(X, y, noise_level=0.03, n_copies=3):
    """
    Cria n_copies cópias ruidosas de X e devolve X_aug, y_aug.
    """
    X_aug = [X]          # lista com o conjunto original
    y_aug = [y]
    for _ in range(n_copies):
        noise = np.random.normal(0, noise_level, X.shape)
        X_aug.append(X + noise)   # nova cópia + ruído
        y_aug.append(y)
    return np.vstack(X_aug), np.hstack(y_aug)

def generate_synthetic_regression(X, y, n_samples=2000):
    gmm = GaussianMixture(n_components=3, random_state=0).fit(X)
    X_syn, _ = gmm.sample(n_samples)
    y_syn = np.random.choice(y, n_samples, replace=True)
    return np.vstack([X, X_syn]), np.hstack([y, y_syn])

def generate_synthetic_classification(X, y, n_per_class=500):
    X_all, y_all = [X], [y]
    for cls in np.unique(y):
        X_cls = X[y == cls]
        gmm = GaussianMixture(n_components=2, random_state=0).fit(X_cls)
        X_gen, _ = gmm.sample(n_per_class)
        X_all.append(X_gen)
        y_all.append(np.full(n_per_class, cls))
    return np.vstack(X_all), np.hstack(y_all)

# --- conjuntos aumentados ---
X_jit_reg, y_jit_reg = augment_jitter(X_scaled, y_reg)
X_jit_clf, y_jit_clf = augment_jitter(X_scaled, y_clf)
X_smote, y_smote = SMOTE(random_state=42).fit_resample(X_scaled, y_clf)
X_gmm_reg, y_gmm_reg = generate_synthetic_regression(X_scaled, y_reg)
X_gmm_clf, y_gmm_clf = generate_synthetic_classification(X_scaled, y_clf)

# ---------------------------------------------------------------------
# 4. MODELOS BÁSICOS + NOVOS
# ---------------------------------------------------------------------
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, KFold
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    HistGradientBoostingRegressor, HistGradientBoostingClassifier,
    VotingClassifier, VotingRegressor
)
from sklearn.linear_model import LinearRegression, LogisticRegression

# XGBoost é opcional
try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# ---------- 4.1 Ajuste fino: Random Forest ---------------------------
rf_reg = RandomForestRegressor(n_jobs=-1, random_state=42)
rf_clf = RandomForestClassifier(n_jobs=-1, random_state=42)

param_dist_reg = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [10, 15, 20, 25, 30, None],
    "min_samples_split": [2, 5, 10, 15],
    "max_features": ["sqrt", "log2", None],
    "min_samples_leaf": [1, 2, 4, 8],
    "bootstrap": [True, False]
}

param_dist_clf = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [10, 15, 20, 25, 30, None],
    "min_samples_split": [2, 5, 10, 15],
    "max_features": ["sqrt", "log2", None],
    "min_samples_leaf": [1, 2, 4, 8],
    "bootstrap": [True, False]
}

tuner_reg = RandomizedSearchCV(
    rf_reg, param_distributions=param_dist_reg,
    n_iter=20, cv=5, scoring="neg_root_mean_squared_error",
    n_jobs=-1, random_state=42
).fit(X_jit_reg, y_jit_reg)

tuner_clf = RandomizedSearchCV(
    rf_clf, param_distributions=param_dist_clf,
    n_iter=20, cv=5, scoring="accuracy",
    n_jobs=-1, random_state=42
).fit(X_smote, y_smote)

best_rf_reg = tuner_reg.best_estimator_
best_rf_clf = tuner_clf.best_estimator_

print("Melhor RF Regressor  ->", tuner_reg.best_params_)
print("Melhor RF Classifier ->", tuner_clf.best_params_)

# Avaliação do melhor RandomForest já treinado (regressor)
score_reg = cross_val_score(best_rf_reg, X_jit_reg, y_jit_reg, cv=kf, scoring="neg_root_mean_squared_error", n_jobs=-1)
print(f"\n🏁 Melhor RFRegressor (ajustado): RMSE médio {-score_reg.mean():.4f}")

# Avaliação do melhor RandomForest já treinado (classificador)
score_clf = cross_val_score(best_rf_clf, X_smote, y_smote, cv=kf, scoring="accuracy", n_jobs=-1)
print(f"🏁 Melhor RFClassifier (ajustado): Acurácia média {score_clf.mean():.4f}")


from sklearn.preprocessing import LabelEncoder

# Codifica as classes para começar em 0 (apenas para XGBoost)
le = LabelEncoder()
y_clf_xgb = le.fit_transform(y_clf)
y_jit_clf_xgb = le.transform(y_jit_clf)
y_smote_xgb = le.transform(y_smote)


# ---------- 4.2 Outros modelos --------------------------------------
regressors = [
    ("RF", RandomForestRegressor(**tuner_reg.best_params_)),
    ("HGBR", HistGradientBoostingRegressor()),
    ("DT_10", DecisionTreeRegressor(max_depth=10)),
    ("Linear", LinearRegression())
]
classifiers = [
    ("RF", RandomForestClassifier(**tuner_clf.best_params_)),
    ("HGBC", HistGradientBoostingClassifier()),
    ("DT_10", DecisionTreeClassifier(max_depth=10)),
    ("LogReg", LogisticRegression(max_iter=1000))
]

if HAS_XGB:
    classifiers.append(("XGBC", XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8)))

# ---------- 4.3 Ensembles (Voting) -----------------------------------
voting_reg = VotingRegressor(regressors)
voting_clf = VotingClassifier(classifiers, voting="soft")

# Só agora, depois de criar, adicione os ensembles às listas para avaliação
regressors_com_ensemble = regressors + [("Voting", voting_reg)]
classifiers_com_ensemble = classifiers + [("Voting", voting_clf)]

# ---------------------------------------------------------------------
# 5. AVALIAÇÃO + VISUALIZAÇÃO
# ---------------------------------------------------------------------
def avaliar(nome, X, y, modelos, scoring, y_xgb=None):
    print(f"\n🏁 {nome}")
    for label, model in modelos:
        # Se for XGBoost, use os rótulos codificados
        if label == "XGBC" and y_xgb is not None:
            score = cross_val_score(model, X, y_xgb, cv=kf, scoring=scoring, n_jobs=-1)
        else:
            score = cross_val_score(model, X, y, cv=kf, scoring=scoring, n_jobs=-1)
        metric = -score.mean() if scoring.startswith("neg") else score.mean()
        print(f"{label:<10s}: {metric:.4f}")

# ----------------- Avaliação de Regressão ---------------------------
avaliar("Regressão (Jitter)  - RMSE", X_jit_reg, y_jit_reg,
        regressors_com_ensemble, scoring="neg_root_mean_squared_error")

avaliar("Regressão (GMM)     - RMSE", X_gmm_reg, y_gmm_reg,
        regressors_com_ensemble, scoring="neg_root_mean_squared_error")

# ----------------- Avaliação de Classificação -----------------------
avaliar("Classificação (SMOTE) - Acc", X_smote, y_smote,
        classifiers_com_ensemble, scoring="accuracy", y_xgb=y_smote_xgb)

avaliar("Classificação (Jitter) - Acc", X_jit_clf, y_jit_clf,
        classifiers_com_ensemble, scoring="accuracy", y_xgb=y_jit_clf_xgb)

# ---------- 5.1 Confusion Matrix do melhor classificador ------------
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

best_clf = best_rf_clf  # ou voting_clf se preferir
best_clf.fit(X_smote, y_smote)
y_pred = best_clf.predict(X_scaled)
cm = confusion_matrix(y_clf, y_pred, labels=np.unique(y_clf))

fig, ax = plt.subplots(figsize=(5, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=np.unique(y_clf))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("Confusion Matrix – melhor classificador")
plt.tight_layout()
plt.show()

import joblib

# Caminhos dos modelos
rf_reg_path = os.path.join(ROOT, "best_rf_reg.joblib")
xgb_clf_path = os.path.join(ROOT, "best_xgb_clf.joblib")
scaler_path = os.path.join(ROOT, "scaler.joblib")

# # Treinar e salvar apenas se não existirem
# if not (os.path.exists(rf_reg_path) and os.path.exists(xgb_clf_path) and os.path.exists(scaler_path)):
#     # Treinar o XGBoost final com todos os dados (SMOTE)
#     if HAS_XGB:
#         final_xgb_clf = XGBClassifier(
#             n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8
#         )
#         final_xgb_clf.fit(X_smote, y_smote_xgb)
#         joblib.dump(final_xgb_clf, xgb_clf_path)

#     joblib.dump(best_rf_reg, rf_reg_path)
#     joblib.dump(scaler, scaler_path)
#     print("✅ Modelos treinados e salvos.")
# else:
#     print("✅ Modelos já existem, pulando treinamento.")