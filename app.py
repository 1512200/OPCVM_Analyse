import os
from pathlib import Path

import pandas as pd
import streamlit as st

# Optional (visualisations)
import plotly.express as px
import plotly.graph_objects as go


# ================================
# PAGE CONFIG (must be first Streamlit command)
# ================================
st.set_page_config(
    page_title="OPCVM Intelligence Platform",
    layout="wide",
)


# ================================
# STYLE (Accueil centré)
# ================================
st.markdown(
    """
    <style>
      .centered { text-align: center; }
      .project-title { font-size: 42px; font-weight: 800; margin-bottom: 10px; }
      .project-desc { font-size: 18px; color: #bfbfbf; max-width: 950px; margin: 0 auto; }
      .small-muted { color: #9aa4b2; font-size: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ================================
# DATA LOADING
# ================================
SCORES_DIR = "scoring_par_categorie"  # folder that contains scored_*.csv and/or opcvm_scored_all.csv


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
# SIDEBAR (logo + navigation)
# ================================
logo_candidates = ["logo.png", "logo.jpg", "logo.jpeg"]
logo_path = next((p for p in logo_candidates if Path(p).exists()), None)

if logo_path:
    st.sidebar.image(logo_path, width=140)

st.sidebar.markdown(
    "<h3 style='text-align:center; margin-top: 0.2rem;'>OPCVM Intelligence</h3>",
    unsafe_allow_html=True,
)
st.sidebar.divider()

page = st.sidebar.radio(
    "📌 Navigation",
    [
        "🏠 Accueil",
        "📊 Tableau des Fonds",
        "⭐ Classement & Recommandation",
        "📈 Dashboard",
        "🤖 Machine Learning (option)",
    ],
)


# ================================
# GUARD: data presence
# ================================
if df_all.empty:
    st.error(
        "Impossible de charger les scores. Vérifie :\n"
        f"- le dossier **{SCORES_DIR}/** existe\n"
        "- il contient **opcvm_scored_all.csv** ou des fichiers **scored_*.csv**\n"
    )
    st.stop()


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

df_all
# ================================
# PAGE 1 — ACCUEIL
# ================================
if page == "🏠 Accueil":
    st.markdown(
        """
        <div class="centered">
          <div class="project-title">Plateforme d’analyse, scoring et recommandation des OPCVM</div>
          <div class="project-desc">
            Cette plateforme permet d’explorer, comparer et classer les fonds OPCVM à partir de scores
            <b>Performance</b>, <b>Risque</b>, <b>RAROC</b> et d’un <b>Score Global</b>.
            Elle propose aussi une recommandation personnalisée selon le profil de l’investisseur.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    n_funds = df_all["fond"].nunique() if "fond" in df_all.columns else len(df_all)
    n_cats = df_all["Catégorie"].nunique() if "Catégorie" in df_all.columns else 0
    best = df_all["score_global"].max() if "score_global" in df_all.columns else None
    best_fund = (
        df_all.loc[df_all["score_global"].idxmax(), "fond"]
        if "score_global" in df_all.columns and "fond" in df_all.columns
        else "—"
    )
    c1.metric("Nombre de fonds", f"{n_funds}")
    c2.metric("Nombre de catégories", f"{n_cats}")
    c3.metric("Meilleur score global", f"{best:.3f}" if best is not None else "—")
    c4.metric("Top fonds", best_fund)

    st.markdown("<br>", unsafe_allow_html=True)
    a, b, c = st.columns(3)
    with a:
        st.info("📊 Tableaux par catégorie\n\nAccès rapide aux scores par fonds.")
    with b:
        st.success("⭐ Classement & recommandation\n\nTop global + profil investisseur.")
    with c:
        st.warning("📈 Dashboard\n\nRisque vs performance (scatter interactif).")

    st.markdown(
        "<p class='small-muted centered'>Données : fichiers CSV scorés (1 ligne par fonds)</p>",
        unsafe_allow_html=True,
    )


# ================================
# PAGE 2 — TABLEAU DES FONDS (1 tableau par catégorie)
# ================================
elif page == "📊 Tableau des Fonds":
    st.title("📊 Tableau des Fonds — Scores par catégorie")
    st.write("Affichage des scores **déjà calculés** (1 ligne par fonds).")

    categories = sorted(df_all["Catégorie"].dropna().unique().tolist())
    selected_cat = st.multiselect("Catégories :", categories, default=categories)

    top_n = st.slider("Nombre de fonds à afficher par catégorie", 5, 200, 30)
    show_expanders = st.toggle("Afficher sous forme d’onglets (expanders)", value=True)

    cols_to_show = [
        "fond",
        "score_performance",
        "score_risque",
        "score_raroc",
        "score_global",
    ]
    cols_to_show = [c for c in cols_to_show if c in df_all.columns]

    st.markdown("### 📄 Résultats")
    for cat in selected_cat:
        cat_df = df_all[df_all["Catégorie"] == cat].copy()
        if "score_global" in cat_df.columns:
            cat_df = cat_df.sort_values("score_global", ascending=False)

        cat_df = fmt_cols(cat_df)

        header = f"📌 Catégorie : {cat}  —  {len(cat_df)} fonds"
        if show_expanders:
            with st.expander(header, expanded=False):
                st.dataframe(cat_df[cols_to_show].head(top_n), use_container_width=True)
        else:
            st.subheader(header)
            st.dataframe(cat_df[cols_to_show].head(top_n), use_container_width=True)
            st.divider()


# ================================
# PAGE 3 — CLASSEMENT & RECOMMANDATION
# ================================
elif page == "⭐ Classement & Recommandation":
    st.title("⭐ Classement & Recommandation")

    st.subheader("🏆 Classement global")
    n = st.slider("Top N (global)", 5, 50, 10)
    cols_rank = [
        "fond",
        "Catégorie",
        "score_global",
        "score_performance",
        "score_risque",
        "score_raroc",
    ]
    cols_rank = [c for c in cols_rank if c in df_all.columns]
    top_global = fmt_cols(df_all.sort_values("score_global", ascending=False).head(n))
    st.dataframe(top_global[cols_rank], use_container_width=True)

    st.divider()

    st.subheader("📌 Classement par catégorie")
    cat = st.selectbox("Choisir une catégorie", sorted(df_all["Catégorie"].unique()))
    df_cat = fmt_cols(df_all[df_all["Catégorie"] == cat].sort_values("score_global", ascending=False))
    st.dataframe(
        df_cat[[c for c in cols_rank if c != "Catégorie"]].head(50),
        use_container_width=True,
    )

    st.divider()

    st.subheader("🎯 Recommandation personnalisée (profil investisseur)")
    profil = st.selectbox("Profil", ["Prudent", "Équilibré", "Dynamique"])
    restrict_cat = st.toggle("Recommander uniquement dans la catégorie sélectionnée", value=False)

    df_base = df_cat.copy() if restrict_cat else df_all.copy()

    if profil == "Prudent":
        weights = {"score_risque": 0.60, "score_performance": 0.20, "score_raroc": 0.20}
        st.info("Profil prudent : priorité à la stabilité (risque faible).")
    elif profil == "Équilibré":
        weights = {"score_risque": 0.40, "score_performance": 0.30, "score_raroc": 0.30}
        st.info("Profil équilibré : compromis rendement / risque.")
    else:
        weights = {"score_risque": 0.20, "score_performance": 0.55, "score_raroc": 0.25}
        st.info("Profil dynamique : priorité à la performance/potentiel.")

    # Compute profil_score from existing scores (rule-based, explainable)
    df_base = df_base.copy()
    df_base["profil_score"] = 0.0
    for k, w in weights.items():
        if k in df_base.columns:
            df_base["profil_score"] += w * df_base[k].fillna(0)

    reco_n = st.slider("Top N recommandé", 3, 20, 5)
    reco = fmt_cols(df_base.sort_values("profil_score", ascending=False).head(reco_n))

    cols_reco = [
        "fond",
        "Catégorie",
        "profil_score",
        "score_global",
        "score_performance",
        "score_risque",
        "score_raroc",
    ]
    cols_reco = [c for c in cols_reco if c in reco.columns]
    st.dataframe(reco[cols_reco], use_container_width=True)

    st.caption(f"Pondérations utilisées : {weights}")


# ================================
# PAGE 4 — DASHBOARD (scatter)
# ================================
elif page == "📈 Dashboard":
    st.title("📈 Dashboard d’analyse")
    st.write("Scatter plot : **x = risque**, **y = performance**, chaque point = un fonds, couleur = catégorie.")

    categories = sorted(df_all["Catégorie"].dropna().unique().tolist())
    selected_cat = st.multiselect("Catégories", categories, default=categories)
    df_f = df_all[df_all["Catégorie"].isin(selected_cat)].copy()

    # Choose axes depending on what you have
    x_options = []
    if "risk_raw" in df_f.columns:
        x_options.append("Risque (risk_raw)")
    if "score_risque" in df_f.columns:
        x_options.append("Score risque (stabilité, 0..1)")
        x_options.append("Risque (1 - score_risque)")

    y_options = []
    if "perf_mean" in df_f.columns:
        y_options.append("Performance (perf_mean)")
    if "score_performance" in df_f.columns:
        y_options.append("Score performance (0..1)")

    x_choice = st.selectbox("Axe X", x_options, index=0)
    y_choice = st.selectbox("Axe Y", y_options, index=0)

    if x_choice == "Risque (risk_raw)":
        x_col = "risk_raw"
        x_label = "Risque (proxy)"
    elif x_choice == "Risque (1 - score_risque)":
        df_f["risk_from_score"] = 1 - df_f["score_risque"]
        x_col = "risk_from_score"
        x_label = "Risque (1 - score_risque)"
    else:
        x_col = "score_risque"
        x_label = "Score risque (stabilité)"

    if y_choice == "Performance (perf_mean)":
        y_col = "perf_mean"
        y_label = "Performance (proxy)"
    else:
        y_col = "score_performance"
        y_label = "Score performance"

    # Optional filter on score_global
    if "score_global" in df_f.columns:
        lo, hi = float(df_f["score_global"].min()), float(df_f["score_global"].max())
        r = st.slider("Filtrer sur score global", lo, hi, (lo, hi))
        df_f = df_f[(df_f["score_global"] >= r[0]) & (df_f["score_global"] <= r[1])]

    # Scatter plot
    hover_cols = [c for c in ["fond", "Catégorie", "score_global", "score_raroc"] if c in df_f.columns]
    fig = px.scatter(
        df_f,
        x=x_col,
        y=y_col,
        color="Catégorie",
        hover_name="fond" if "fond" in df_f.columns else None,
        hover_data=hover_cols,
        size="score_global" if "score_global" in df_f.columns else None,
        title="Risque vs performance (par catégorie)",
        labels={x_col: x_label, y_col: y_label},
    )
    fig.update_layout(height=650)

    # Highlight selected fund
    if "fond" in df_f.columns and df_f["fond"].nunique() > 0:
        fund = st.selectbox("Mettre en évidence un fonds", ["(aucun)"] + sorted(df_f["fond"].unique().tolist()))
        if fund != "(aucun)":
            row = df_f[df_f["fond"] == fund].head(1)
            if not row.empty:
                fig.add_trace(
                    go.Scatter(
                        x=row[x_col],
                        y=row[y_col],
                        mode="markers",
                        marker=dict(size=18, symbol="star"),
                        name=f"Sélection : {fund}",
                        showlegend=True,
                    )
                )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("🔎 Détails d’un fonds")
    if "fond" in df_f.columns and df_f["fond"].nunique() > 0:
        fund2 = st.selectbox("Choisir un fonds", sorted(df_f["fond"].unique().tolist()), key="fund_details")
        row2 = df_all[df_all["fond"] == fund2].head(1)
        if not row2.empty:
            row2 = row2.iloc[0]
            st.json(
                {
                    "Fonds": row2.get("fond"),
                    "Catégorie": row2.get("Catégorie"),
                    "Score Performance": float(row2.get("score_performance")) if pd.notna(row2.get("score_performance")) else None,
                    "Score Risque": float(row2.get("score_risque")) if pd.notna(row2.get("score_risque")) else None,
                    "Score RAROC": float(row2.get("score_raroc")) if pd.notna(row2.get("score_raroc")) else None,
                    "Score Global": float(row2.get("score_global")) if pd.notna(row2.get("score_global")) else None,
                }
            )


# ================================
# PAGE 5 — ML (KMeans) (option)
# ================================
elif page == "🤖 Machine Learning (option)":
    st.title("🤖 Machine Learning — Segmentation des fonds (KMeans)")
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

    st.subheader("📌 Visualisation des clusters")
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
    st.subheader("📊 Synthèse par cluster")
    summary = (
        df_ml.groupby("cluster")[features]
        .mean()
        .round(3)
        .reset_index()
        .sort_values("cluster")
    )
    st.dataframe(summary, use_container_width=True)

    st.divider()
    st.subheader("🎯 Exemple : associer un cluster à un profil")
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
