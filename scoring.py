import os
import re
import numpy as np
import pandas as pd

# =========================
# CONFIG
# =========================
INPUT_DIR = "categories_csv"          # dossier contenant tes fichiers par catégorie
OUTPUT_DIR = "scoring_par_categorie"  # dossier de sortie

DATE_COL = "date"
FUND_COL = "fond"
CATEGORY_COL = "Catégorie"

PERF_COLS = ["1W (%)", "6M (%)", "1A (%)", "2A (%)", "3A (%)", "5A (%)"]

# Pondérations pour performance composite (somme = 1)
WEIGHTS = {
    "1W (%)": 0.10,
    "6M (%)": 0.20,
    "1A (%)": 0.20,
    "2A (%)": 0.20,
    "3A (%)": 0.15,
    "5A (%)": 0.15,
}

# Pondérations score global (somme = 1)
W_PERF = 0.4
W_RISK = 0.3
W_RAROC = 0.3

NORMALIZATION = "percentile"  # "percentile" ou "minmax"
EPS = 1e-9


# =========================
# HELPERS
# =========================
def safe_numeric(s):
    return pd.to_numeric(
        s.astype(str).str.replace(",", ".", regex=False).str.replace("%", "", regex=False),
        errors="coerce"
    )

def normalize(series, higher_is_better=True, method="percentile"):
    x = series.copy()
    if method == "percentile":
        score = x.rank(pct=True, na_option="keep")  # 0..1
    else:
        minv, maxv = np.nanmin(x.values), np.nanmax(x.values)
        if np.isclose(maxv, minv) or np.isnan(minv) or np.isnan(maxv):
            score = pd.Series(np.nan, index=x.index)
        else:
            score = (x - minv) / (maxv - minv)

    if not higher_is_better:
        score = 1 - score
    return score

def slugify(text: str) -> str:
    text = str(text).strip().lower()
    text = text.replace("&", "and")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_]+", "", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "categorie_sans_nom"


# =========================
# SCORING POUR UNE CATEGORIE (un fichier)
# =========================
def compute_scoring_for_one_category(df_cat: pd.DataFrame) -> pd.DataFrame:
    required = [DATE_COL, FUND_COL, CATEGORY_COL] + PERF_COLS
    missing = [c for c in required if c not in df_cat.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le fichier: {missing}")

    df = df_cat.copy()

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    for c in PERF_COLS:
        df[c] = safe_numeric(df[c])

    # Performance composite par ligne (par date)
    w = pd.Series(WEIGHTS)
    w = w / w.sum()
    df["perf_raw"] = df[PERF_COLS].mul(w[PERF_COLS], axis=1).sum(axis=1, skipna=True)

    # Agrégation par fonds sur la période (oct+nov)
    agg = (
        df.groupby([CATEGORY_COL, FUND_COL], as_index=False)
          .agg(
              perf_mean=("perf_raw", "mean"),
              risk_raw=("perf_raw", "std"),
              nb_dates=(DATE_COL, "nunique")
          )
    )

    agg["risk_raw"] = agg["risk_raw"].fillna(0.0)
    agg["raroc_raw"] = agg["perf_mean"] / (agg["risk_raw"] + EPS)

    # Normalisation (dans cette catégorie)
    agg["score_performance"] = normalize(agg["perf_mean"], higher_is_better=True, method=NORMALIZATION)
    agg["score_risque"] = normalize(agg["risk_raw"], higher_is_better=False, method=NORMALIZATION)
    agg["score_raroc"] = normalize(agg["raroc_raw"], higher_is_better=True, method=NORMALIZATION)

    agg["score_global"] = (
        W_PERF * agg["score_performance"] +
        W_RISK * agg["score_risque"] +
        W_RAROC * agg["score_raroc"]
    )

    cols = [
        CATEGORY_COL, FUND_COL, "nb_dates",
        "score_performance", "score_risque", "score_raroc", "score_global",
        "perf_mean", "risk_raw", "raroc_raw"
    ]
    return agg[cols].sort_values("score_global", ascending=False)


# =========================
# MAIN : LOOP SUR LES FICHIERS
# =========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    scored_all = []

    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".csv") and not f.startswith("_")]
    if not files:
        raise FileNotFoundError(f"Aucun CSV trouvé dans {INPUT_DIR}/")

    for fname in files:
        path = os.path.join(INPUT_DIR, fname)
        df_cat = pd.read_csv(path)

        # (Option) garantir que la catégorie est bien remplie même si tu l'as perdue
        if CATEGORY_COL not in df_cat.columns:
            # on déduit du nom de fichier
            inferred_cat = os.path.splitext(fname)[0]
            df_cat[CATEGORY_COL] = inferred_cat

        scored = compute_scoring_for_one_category(df_cat)

        # nom de catégorie (dans le fichier)
        cat_name = str(scored[CATEGORY_COL].iloc[0])
        out_path = os.path.join(OUTPUT_DIR, f"scored_{slugify(cat_name)}.csv")
        scored.to_csv(out_path, index=False, encoding="utf-8-sig")

        scored_all.append(scored)
        print(f"✅ {fname} -> {os.path.basename(out_path)} ({len(scored)} fonds)")

    # fichier global
    all_df = pd.concat(scored_all, ignore_index=True)
    all_df.to_csv(os.path.join(OUTPUT_DIR, "opcvm_scored_all.csv"),
                  index=False, encoding="utf-8-sig")

    print(f"\n Terminé. Résultats dans: {OUTPUT_DIR}/")
    print(" - opcvm_scored_all.csv")
    print(" - scored_<categorie>.csv (un par catégorie)")


if __name__ == "__main__":
    main()
