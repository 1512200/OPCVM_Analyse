import os
import re
import pandas as pd

# =========================
# CONFIG
# =========================
INPUT_CSV = "DATA OPCVM/data.csv"          # <-- ton fichier source
CATEGORY_COL = "Catégorie"       # <-- nom exact de la colonne catégorie
OUTPUT_DIR = "categories_csv"    # dossier de sortie

# =========================
# HELPERS
# =========================
def slugify(text: str) -> str:
    """Transforme un texte en nom de fichier propre."""
    text = str(text).strip()
    text = text.lower()
    text = text.replace("&", "and")
    text = re.sub(r"\s+", "_", text)          # espaces -> _
    text = re.sub(r"[^a-z0-9_]+", "", text)   # enlever caractères spéciaux
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "categorie_sans_nom"

# =========================
# MAIN
# =========================
def split_by_category():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    if CATEGORY_COL not in df.columns:
        raise ValueError(
            f"Colonne '{CATEGORY_COL}' introuvable. Colonnes disponibles: {list(df.columns)}"
        )

    # Nettoyage léger
    df[CATEGORY_COL] = df[CATEGORY_COL].astype(str).str.strip()

    # Drop catégories vides si besoin
    df = df[df[CATEGORY_COL].notna() & (df[CATEGORY_COL] != "")].copy()

    summary = []

    for cat, grp in df.groupby(CATEGORY_COL):
        filename = f"{slugify(cat)}.csv"
        out_path = os.path.join(OUTPUT_DIR, filename)

        grp.to_csv(out_path, index=False, encoding="utf-8-sig")
        summary.append({
            "categorie": cat,
            "nb_lignes": len(grp),
            "fichier": out_path
        })

    # Fichier résumé
    summary_df = pd.DataFrame(summary).sort_values("nb_lignes", ascending=False)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "_resume_categories.csv"),
                      index=False, encoding="utf-8-sig")

    print(f"✅ Terminé. {len(summary_df)} fichiers créés dans '{OUTPUT_DIR}/'")
    print(summary_df.head(10).to_string(index=False))

if __name__ == "__main__":
    split_by_category()
