import streamlit as st
import pandas as pd
import os
import plotly.express as px
from pathlib import Path

# ================================
# PAGE CONFIG (TOUJOURS EN PREMIER)
# ================================
st.set_page_config(
    page_title="OPCVM Intelligence Platform",
    layout="wide"
)
SCORES_DIR = "scoring_par_categorie"
def fmt_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with rounded score columns for nicer display."""
    df2 = df.copy()
    for c in ["score_performance", "score_risque", "score_raroc", "score_global", "profil_score"]:
        if c in df2.columns:
            df2[c] = df2[c].round(3)
    for c in ["perf_mean", "risk_raw", "raroc_raw"]:
        if c in df2.columns:
            df2[c] = df2[c].round(4)
    return df2
@st.cache_data(show_spinner=False)
def load_scored_data(scores_dir: str) -> pd.DataFrame:
    """Load all scored funds. Expected: 1 row per fund with score columns."""
    scores_path = Path(scores_dir)
    if not scores_path.exists():
        return pd.DataFrame()

    # Prefer a global file if available
    global_path = scores_path / "opcvm_scored_all.csv"
    if global_path.exists():
        df = pd.read_csv(global_path)
    else:
        files = sorted([p for p in scores_path.glob("scored_*.csv") if p.is_file()])
        if not files:
            return pd.DataFrame()
        df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)

    # Ensure expected columns exist
    if "Catégorie" not in df.columns and "Categorie" in df.columns:
        df = df.rename(columns={"Categorie": "Catégorie"})

    # Numeric columns (convert safely)
    numeric_cols = [
        "score_performance",
        "score_risque",
        "score_raroc",
        "score_global",
        "perf_mean",
        "risk_raw",
        "raroc_raw",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Basic cleanup
    if "fond" in df.columns:
        df["fond"] = df["fond"].astype(str).str.strip()
    if "Catégorie" in df.columns:
        df["Catégorie"] = df["Catégorie"].astype(str).str.strip()

    return df


df_all = load_scored_data(SCORES_DIR)
# ================================
# STYLE (CENTRAGE TITRE & TEXTE)
# ================================
st.markdown(
    """
    <style>
    .centered {
        text-align: center;
    }
    .project-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 12px;
        color: #8d1813;
    }
    .project-desc {
        font-size: 18px;
        color: black;
        max-width: 900px;
        margin: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ================================
# LOAD DATA
# ================================
@st.cache_data
def load_data():
    return pd.read_csv("opcvm_scored.csv")

df = load_data()

# ================================
# SIDEBAR NAVIGATION
# ================================

st.sidebar.image(
    "logo.png",
    width=140  # ← ajuste ici (120 / 130 / 150)
)

st.sidebar.divider()

page = st.sidebar.radio(
    " Navigation",
    [
        " Accueil",
        " Tableau des Fonds",
        " Classement & Recommandation",
        " Analyse par Fonds",
        " Segmentation des fonds "
    ]
)


# ===========================================================
# 0) PAGE : ACCUEIL
# ===========================================================
if page == " Accueil":

    st.markdown(
        """
        <div class="centered">
            <div class="project-title">
                Plateforme d’analyse, scoring et recommandation des OPCVM
            </div>
            <div class="project-desc">
                Cette plateforme permet d’analyser, comparer et classer les fonds OPCVM
                à partir d’indicateurs de performance, de risque et de RAROC.
                Elle propose également des recommandations personnalisées
                selon le profil de l’investisseur.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br><br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.info(" Exploration et filtrage des fonds OPCVM")
    with c2:
        st.success(" Classement global et recommandations adaptées")
    with c3:
        st.warning(" Analyse détaillée et comparative par fonds")       

# ===========================================================
# 1) PAGE : TABLEAU DES FONDS
# ===========================================================

elif page == " Tableau des Fonds":

    st.title(" Tableau des Fonds OPCVM (Scores par catégorie)")
    

    

    # ---------
    # Charger tous les fichiers scored_*.csv
    # ---------
    if not os.path.exists(SCORES_DIR):
        st.error(f" Dossier introuvable : {SCORES_DIR}")
        st.stop()

    files = sorted([
        f for f in os.listdir(SCORES_DIR)
        if f.lower().endswith(".csv") and f.startswith("scored_")
    ])

    if not files:
        st.warning(f"Aucun fichier trouvé dans {SCORES_DIR}/ (attendu: scored_*.csv).")
        st.stop()

    # Lire tous les fichiers et construire un dict {categorie: dataframe}
    cat_tables = {}
    for f in files:
        path = os.path.join(SCORES_DIR, f)
        tmp = pd.read_csv(path)

        # on récupère le nom catégorie depuis la colonne si elle existe, sinon depuis le nom de fichier
        if "Catégorie" in tmp.columns and tmp["Catégorie"].notna().any():
            cat_name = str(tmp["Catégorie"].dropna().iloc[0])
        else:
            cat_name = f.replace("scored_", "").replace(".csv", "")

        cat_tables[cat_name] = tmp

    # ---------
    # Filtre catégories
    # ---------
    categories = sorted(cat_tables.keys())
    selected_cat = st.multiselect("Catégories :", categories, default=categories)

    # Colonnes à afficher (si présentes)
    preferred_cols = ["fond", "score_performance", "score_risque", "score_raroc", "score_global"]

    

    for cat in selected_cat:
        df_cat = cat_tables.get(cat)
        if df_cat is None or df_cat.empty:
            continue

        st.subheader(f"Catégorie : {cat}")

        cols_to_show = [c for c in preferred_cols if c in df_cat.columns]
        if not cols_to_show:
            st.warning("Colonnes attendues non trouvées dans ce fichier.")
            st.dataframe(df_cat, use_container_width=True)
        else:
            # Trier par score_global si dispo
            if "score_global" in df_cat.columns:
                df_cat = df_cat.sort_values("score_global", ascending=False)

            st.dataframe(df_cat[cols_to_show], use_container_width=True)

        st.divider()

# ===========================================================
# 2) PAGE : CLASSEMENT & RECOMMANDATION
# ===========================================================
elif page == " Classement & Recommandation":

    st.title(" Classement & Recommandation des Fonds")

    SCORES_DIR = "scoring_par_categorie"

    # ---------
    # Charger tous les CSV scorés
    # ---------
    files = [
        f for f in os.listdir(SCORES_DIR)
        if f.lower().endswith(".csv") and f.startswith("scored_")
    ]

    if not files:
        st.error("Aucun fichier de scoring trouvé.")
        st.stop()

    dfs = []
    for f in files:
        df_tmp = pd.read_csv(os.path.join(SCORES_DIR, f))
        dfs.append(df_tmp)

    df_all = pd.concat(dfs, ignore_index=True)

    # ---------
    # 1) Classement global
    # ---------
    st.subheader(" Classement global (Top 10)")
    top10 = df_all.sort_values("score_global", ascending=False).head(10)

    st.dataframe(
        top10[["fond", "Catégorie", "score_global",
               "score_performance", "score_risque", "score_raroc"]],
        use_container_width=True
    )

    st.divider()

    # ---------
    # 2) Classement par catégorie
    # ---------
    st.subheader(" Classement par catégorie")
    cat = st.selectbox(
        "Choisir une catégorie :",
        sorted(df_all["Catégorie"].unique())
    )

    df_cat = (
        df_all[df_all["Catégorie"] == cat]
        .sort_values("score_global", ascending=False)
    )

    st.dataframe(
        df_cat[["fond", "score_global",
                "score_performance", "score_risque", "score_raroc"]],
        use_container_width=True
    )

    st.divider()

    # ---------
    # 3) Recommandation personnalisée
    # ---------
    st.subheader(" Recommandation personnalisée")

    profil = st.selectbox(
        "Choisissez votre profil investisseur",
        ["Prudent", "Équilibré", "Dynamique"]
    )

    df_local = df_all.copy()

    if profil == "Prudent":
        st.info("Profil prudent : priorité à la stabilité et au risque faible.")
        df_local["profil_score"] = (
            0.60 * df_local["score_risque"] +
            0.20 * df_local["score_performance"] +
            0.20 * df_local["score_raroc"]
        )

    elif profil == "Équilibré":
        st.info("Profil équilibré : compromis rendement / risque.")
        df_local["profil_score"] = (
            0.40 * df_local["score_risque"] +
            0.30 * df_local["score_performance"] +
            0.30 * df_local["score_raroc"]
        )

    else:  # Dynamique
        st.info("Profil dynamique : priorité à la performance.")
        df_local["profil_score"] = (
            0.20 * df_local["score_risque"] +
            0.55 * df_local["score_performance"] +
            0.25 * df_local["score_raroc"]
        )

    st.subheader(" Fonds recommandés (Top 5)")
    reco = df_local.sort_values("profil_score", ascending=False).head(5)

    st.dataframe(
        reco[["fond", "Catégorie", "profil_score",
              "score_global", "score_performance", "score_risque", "score_raroc"]],
        use_container_width=True
    )

# ===========================================================
# 3) PAGE : ANALYSE PAR FONDS
# ===========================================================


elif page == " Analyse par Fonds":

    st.title(" Dashboard Analyse OPCVM")
    st.write("Scatter plot : **Risque (x)** vs **Performance (y)**, chaque point = un fonds, couleur = catégorie.")

    SCORES_DIR = "scoring_par_categorie"

    # Charger tous les CSV scorés
    files = sorted([
        f for f in os.listdir(SCORES_DIR)
        if f.lower().endswith(".csv") and f.startswith("scored_")
    ])

    if not files:
        st.error(f"Aucun fichier 'scored_*.csv' trouvé dans {SCORES_DIR}/")
        st.stop()

    df_all = pd.concat([pd.read_csv(os.path.join(SCORES_DIR, f)) for f in files], ignore_index=True)

    # Vérifier colonnes nécessaires
    needed = {"fond", "Catégorie", "score_risque", "score_performance", "score_global"}
    if not needed.issubset(df_all.columns):
        st.error(f"Colonnes manquantes. Colonnes requises: {sorted(list(needed))}")
        st.stop()

    # -------------------
    # Filtres
    # -------------------
    st.subheader(" Filtres")

    categories = sorted(df_all["Catégorie"].dropna().unique().tolist())
    selected_cat = st.multiselect("Catégories :", categories, default=categories)

    df_filt = df_all[df_all["Catégorie"].isin(selected_cat)].copy()

    # Option: filtre sur score_global
    min_score = float(df_filt["score_global"].min())
    max_score = float(df_filt["score_global"].max())
    score_range = st.slider("Filtrer sur score global :", min_score, max_score, (min_score, max_score))
    df_filt = df_filt[(df_filt["score_global"] >= score_range[0]) & (df_filt["score_global"] <= score_range[1])]

    st.divider()

    # -------------------
    # Scatter plot
    # -------------------
    st.subheader(" Scatter Plot : Risque vs Performance")

    fig = px.scatter(
        df_filt,
        x="score_risque",
        y="score_performance",
        color="Catégorie",
        hover_data={
            "fond": True,
            "score_global": True,
            "score_risque": ":.3f",
            "score_performance": ":.3f",
            "score_raroc": ":.3f" if "score_raroc" in df_filt.columns else False,
            "Catégorie": True
        },
        title="Risque (x) vs Performance (y) — Couleur : Catégorie"
    )

    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # -------------------
    # Détails d’un fonds
    # -------------------
    st.subheader(" Détails d’un fonds")

    fund = st.selectbox("Choisir un fonds :", sorted(df_filt["fond"].unique().tolist()))
    row = df_all[df_all["fond"] == fund].iloc[0]

    st.json({
        "Fonds": row["fond"],
        "Catégorie": row["Catégorie"],
        "Score Performance": float(row["score_performance"]),
        "Score Risque": float(row["score_risque"]),
        "Score RAROC": float(row["score_raroc"]) if "score_raroc" in df_all.columns else None,
        "Score Global": float(row["score_global"])
    })


# ===========================================================
# 4) PAGE : MACHINE LEARNING         
# ===========================================================
elif page == " Segmentation des fonds ":
    st.title(" Machine Learning — Segmentation des fonds (KMeans)")
    st.write(
        "Cette partie est **optionnelle** : elle segmente les fonds en groupes (styles) sans labels. "
        "C’est utile pour enrichir l’analyse et l’interprétation."
    )

    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
    except Exception:
        st.warning(
            "scikit-learn n’est pas installé dans l’environnement. "
            "Ajoute `scikit-learn` dans requirements.txt pour activer cette page."
        )
        st.stop()

    features = [c for c in ["score_performance", "score_risque", "score_raroc", "score_global"] if c in df_all.columns]
    if len(features) < 2:
        st.error("Pas assez de colonnes de scores pour faire du clustering.")
        st.stop()

    df_ml = df_all.dropna(subset=features).copy()
    k = st.slider("Nombre de clusters (K)", 2, 6, 3)

    X = df_ml[features].values
    Xs = StandardScaler().fit_transform(X)
    model = KMeans(n_clusters=k, random_state=42, n_init="auto")
    df_ml["cluster"] = model.fit_predict(Xs).astype(str)

    st.subheader(" Visualisation des clusters")
    # Use the same scatter defaults as dashboard
    x_col = "score_risque" if "score_risque" in df_ml.columns else features[0]
    y_col = "score_performance" if "score_performance" in df_ml.columns else features[1]

    fig = px.scatter(
        df_ml,
        x=x_col,
        y=y_col,
        color="cluster",
        hover_name="fond" if "fond" in df_ml.columns else None,
        hover_data=[c for c in ["Catégorie", "score_global", "score_raroc"] if c in df_ml.columns],
        title="Segmentation KMeans (couleur = cluster)",
    )
    fig.update_layout(height=650)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader(" Synthèse par cluster")
    summary = (
        df_ml.groupby("cluster")[features]
        .mean()
        .round(3)
        .reset_index()
        .sort_values("cluster")
    )
    st.dataframe(summary, use_container_width=True)

    st.divider()
    st.subheader(" Exemple : associer un cluster à un profil")
    if "score_risque" in df_ml.columns and "score_performance" in df_ml.columns:
        # choose clusters based on mean scores
        cluster_prudent = summary.sort_values("score_risque", ascending=False).iloc[0]["cluster"]
        cluster_dynamique = summary.sort_values("score_performance", ascending=False).iloc[0]["cluster"]

        st.info(
            f"Suggestion (heuristique) :\n"
            f"- **Prudent** → cluster **{cluster_prudent}** (stabilité moyenne la plus élevée)\n"
            f"- **Dynamique** → cluster **{cluster_dynamique}** (performance moyenne la plus élevée)"
        )

        chosen = st.selectbox("Afficher les meilleurs fonds d’un cluster", sorted(df_ml["cluster"].unique()))
        df_c = df_ml[df_ml["cluster"] == chosen].copy()
        if "score_global" in df_c.columns:
            df_c = df_c.sort_values("score_global", ascending=False)
        st.dataframe(fmt_cols(df_c[["fond", "Catégorie", "cluster"] + features].head(20)), use_container_width=True)
