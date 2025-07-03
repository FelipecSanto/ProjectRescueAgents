import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
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

# ------------------ RandomForestRegressor com jitter ---------------------
def augment_jitter(X, y, noise_level=0.03, n_copies=3):
    X_aug = [X]
    y_aug = [y]
    for _ in range(n_copies):
        noise = np.random.normal(0, noise_level, X.shape)
        X_aug.append(X + noise)
        y_aug.append(y)
    return np.vstack(X_aug), np.hstack(y_aug)

X_jit_reg, y_jit_reg = augment_jitter(X_scaled, y_reg)

rf_reg = RandomForestRegressor(
    n_estimators=300,
    max_depth=30,
    min_samples_split=5,
    n_jobs=-1,
    random_state=42
)
rf_reg.fit(X_jit_reg, y_jit_reg)

# ------------------ XGBClassifier com SMOTE -----------------------------
le = LabelEncoder()
y_clf_enc = le.fit_transform(y_clf)
X_smote, y_smote = SMOTE(random_state=42).fit_resample(X_scaled, y_clf_enc)

xgb_clf = XGBClassifier(
    n_estimators=400,
    learning_rate=0.03,
    max_depth=7,
    subsample=0.8,
    n_jobs=-1,
    random_state=42
)
xgb_clf.fit(X_smote, y_smote)

# ------------------ Salva modelos e scaler ------------------------------
joblib.dump(rf_reg, os.path.join(ROOT, "best_rf_reg.joblib"))
joblib.dump(xgb_clf, os.path.join(ROOT, "best_xgb_clf.joblib"))
joblib.dump(scaler, os.path.join(ROOT, "scaler.joblib"))
print("✅ Modelos do Rescuer treinados e salvos!")