import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier
import joblib

# Caminho dos dados
ROOT = os.path.dirname(__file__)
csv_path = os.path.join(ROOT, "data_4000v", "env_vital_signals.csv")
df = pd.read_csv(csv_path)

# Feature engineering
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

# Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Treina apenas os modelos usados pelo Rescuer
rf_reg = RandomForestRegressor(n_estimators=300, min_samples_split=5, max_features=None, max_depth=20, random_state=42, n_jobs=-1)
rf_reg.fit(X_scaled, y_reg)

le = LabelEncoder()
y_clf_xgb = le.fit_transform(y_clf)
xgb_clf = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, random_state=42, n_jobs=-1)
xgb_clf.fit(X_scaled, y_clf_xgb)

# Salva os modelos e scaler
joblib.dump(rf_reg, os.path.join(ROOT, "best_rf_reg.joblib"))
joblib.dump(xgb_clf, os.path.join(ROOT, "best_xgb_clf.joblib"))
joblib.dump(scaler, os.path.join(ROOT, "scaler.joblib"))
print("✅ Modelos do Rescuer treinados e salvos!")