import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ============================
# 1. Load CSV
# ============================
df = pd.read_csv("DATA OPCVM/data.csv")

# ============================
# 2. Normalisation helper
# ============================
def normalize(series):
    scaler = MinMaxScaler()
    values = scaler.fit_transform(series.values.reshape(-1, 1))
    return values.flatten()

# ============================
# 3. Score de Performance
# ============================

df["perf_1A_norm"] = normalize(df["1A (%)"])
df["perf_3A_norm"] = normalize(df["3A (%)"])
df["perf_5A_norm"] = normalize(df["5A (%)"])

df["score_performance"] = (
    0.5 * df["perf_1A_norm"] +
    0.3 * df["perf_3A_norm"] +
    0.2 * df["perf_5A_norm"]
)

# ============================
# 4. Score de Risque
# ============================

df["volatilite"] = abs(df["1W (%)"]) + abs(df["6M (%)"]) / 10
df["vola_norm"] = normalize(df["volatilite"])

df["score_risque"] = 1 - df["vola_norm"]   # faible risque → score élevé

# ============================
# 5. Score RAROC simplifié
# ============================

df["raroc_raw"] = (df["1A (%)"] + df["3A (%)"] + df["5A (%)"]) / (abs(df["1W (%)"]) + 0.1)
df["raroc_norm"] = normalize(df["raroc_raw"])

df["score_raroc"] = df["raroc_norm"]

# ============================
# 6. Score Global (Composite)
# ============================

df["score_global"] = (
    0.40 * df["score_performance"] +
    0.30 * df["score_risque"] +
    0.30 * df["score_raroc"]
)

# ============================
# 7. Classement final
# ============================
df = df.sort_values("score_global", ascending=False)

# ============================
# 8. Export CSV final
# ============================
df.to_csv("opcvm_scored.csv", index=False)

print(" Scoring terminé ! Fichier généré : opcvm_scored.csv")
