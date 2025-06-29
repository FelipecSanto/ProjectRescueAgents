import pandas as pd
import numpy as np
from joblib import load  # se quiser salvar/carregar depois
from sklearn.preprocessing import StandardScaler
import os

# --- 1. Carregar dataset de 800 vítimas ---
ROOT = os.path.dirname(__file__)
csv_path = os.path.join(ROOT, "data_800v", "env_vital_signals.csv")
df_800 = pd.read_csv(csv_path)

# --- 2. Recriar as features adicionais (iguais às do treina_modelos.py) ---
df_800["qPA_pulso_ratio"] = df_800["qPA"] / (df_800["pulso"].replace(0, np.nan))
df_800["pulso_freq_prod"] = df_800["pulso"] * df_800["freq_resp"]
df_800["qPA_minus_pulso"] = df_800["qPA"] - df_800["pulso"]

feature_cols = [
    "qPA", "pulso", "freq_resp",
    "qPA_pulso_ratio", "pulso_freq_prod", "qPA_minus_pulso"
]

X_800 = df_800[feature_cols].fillna(0).values

# --- 3. Usar o mesmo scaler usado em treina_modelos.py ---
# Você pode reutilizar o `scaler` da memória, ou salvar no final do treino: joblib.dump(scaler, 'scaler.joblib')
# Aqui consideramos que ainda está em memória:
from joblib import load

scaler = load("scaler.joblib")
best_rf_reg = load("best_rf_reg.joblib")
best_xgb_clf = load("best_xgb_clf.joblib")

X_800_scaled = scaler.transform(X_800)

# --- 4. Aplicar os melhores modelos ---
y_pred_reg = best_rf_reg.predict(X_800_scaled)
y_pred_clf = best_xgb_clf.predict(X_800_scaled)

# --- 5. Anexar as predições ao DataFrame ---
df_result_800 = df_800.copy()
df_result_800["grav_pred"] = y_pred_reg
df_result_800["classe_pred"] = y_pred_clf

# --- 6. Salvar para uso posterior em clusterização, etc ---
df_result_800.to_csv("resultados_800v.csv", index=False)

print("✅ Predições salvas em resultados_800v.csv")
