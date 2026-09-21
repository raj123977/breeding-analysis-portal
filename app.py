import io
import itertools
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial.distance import pdist, squareform
from scipy.stats import linregress
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
import streamlit as st

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Predictive Breeding & Parent Selection Portal",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🌾 Predictive Breeding & Parent Selection Portal")
st.markdown(
    "Decision Support System for **Genomic/Phenotypic Breeding Value Estimation"
    " (EBV/GEBV)**, **Cross Performance Prediction ($P_1 \\times P_2$)**,"
    " **Heterotic Diversity Clustering**, and **Multi-Environment GxE"
    " Stability**."
)


# ==========================================
# HELPER: GENERATE SAMPLE EXCEL TEMPLATE
# ==========================================
def generate_sample_template():
  np.random.seed(42)
  lines = [f"Line_{i:02d}" for i in range(1, 9)]
  testers = [f"Tester_{j:02d}" for j in range(1, 4)]
  envs = ["Env_North", "Env_South", "Env_Dry"]
  reps = [1, 2]

  data = []
  for env in envs:
    for rep in reps:
      for line in lines:
        for tester in testers:
          yield_val = round(np.random.normal(6.5, 1.2), 2)
          height_val = round(np.random.normal(110, 15), 1)
          days_flower = int(np.random.normal(65, 4))

          # Simulated SNP markers (0, 1, 2)
          m1 = np.random.choice([0, 1, 2], p=[0.25, 0.5, 0.25])
          m2 = np.random.choice([0, 1, 2], p=[0.3, 0.4, 0.3])
          m3 = np.random.choice([0, 1, 2], p=[0.2, 0.6, 0.2])
          m4 = np.random.choice([0, 1, 2], p=[0.4, 0.2, 0.4])

          data.append({
              "Genotype_ID": f"{line} × {tester}",
              "Female_Line": line,
              "Male_Tester": tester,
              "Location": env,
              "Replication": rep,
              "Grain_Yield": yield_val,
              "Plant_Height": height_val,
              "Days_To_Flower": days_flower,
              "M_Marker1": m1,
              "M_Marker2": m2,
              "M_Marker3": m3,
              "M_Marker4": m4,
          })

  sample_df = pd.DataFrame(data)
  buffer = io.BytesIO()
  with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
    sample_df.to_excel(writer, sheet_name="Trial_Data", index=False)
  return buffer.getvalue()


# ==========================================
# SIDEBAR CONTROL PANEL
# ==========================================
st.sidebar.header("📁 Data Input & Template")

template_bytes = generate_sample_template()
st.sidebar.download_button(
    label="📥 Download Predictive Breeding Excel Template",
    data=template_bytes,
    file_name="predictive_breeding_template.xlsx",
    mime="application/vnd.ms-excel",
    help="Download a formatted sample Excel dataset with trial factors, quantitative traits, and SNP markers.",
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Your Trial Dataset (.xlsx or .csv)", type=["xlsx", "csv"]
)


@st.cache_data
def load_data(file):
  if file.name.endswith(".xlsx"):
    return pd.read_excel(file)
  return pd.read_csv(file)


if uploaded_file is not None:
  try:
    df = load_data(uploaded_file)
    st.sidebar.success("Dataset loaded successfully!")

    cols = [str(c) for c in df.columns.tolist()]

    def get_default_index(target_names, col_list, fallback=0):
      for name in target_names:
        for idx, col in enumerate(col_list):
          if name.lower() in col.lower():
            return idx
      return fallback

    st.sidebar.header("⚙️ Column Mapping")

    # 1. Genotype ID Option
    geno_idx = get_default_index(
        ["genotype", "entry", "hybrid", "variety"], cols, fallback=None
    )
    genotype_options = ["Auto-generate (Female × Male)"] + cols
    selected_geno_idx = (
        genotype_options.index(cols[geno_idx]) if geno_idx is not None else 0
    )

    genotype_col_choice = st.sidebar.selectbox(
        "Genotype / Entry ID Column:", genotype_options, index=selected_geno_idx
    )

    # 2. Line & Tester Mapping
    line_idx = get_default_index(
        ["line", "female", "parent1", "mother"], cols, fallback=0
    )
    tester_idx = get_default_index(
        ["tester", "male", "parent2", "father"], cols, fallback=min(1, len(cols) - 1)
    )
    env_idx = get_default_index(
        ["location", "loc", "env", "site"], cols, fallback=min(2, len(cols) - 1)
    )
    rep_idx = get_default_index(
        ["replication", "rep", "block"], cols, fallback=min(3, len(cols) - 1)
    )

    line_col = st.sidebar.selectbox("Female Parent / Line:", cols, index=line_idx)
    tester_col = st.sidebar.selectbox(
        "Male Parent / Tester:", cols, index=tester_idx
    )
    env_col = st.sidebar.selectbox(
        "Location / Environment:", cols, index=env_idx
    )
    rep_col = st.sidebar.selectbox("Replication / Block:", cols, index=rep_idx)

    # Ensure parent identifiers are string types to prevent duplicate index type mismatches
    df[line_col] = df[line_col].astype(str)
    df[tester_col] = df[tester_col].astype(str)

    # Resolve Genotype ID
    if genotype_col_choice == "Auto-generate (Female × Male)":
      df["Genotype_ID"] = (
          df[line_col].astype(str) + " × " + df[tester_col].astype(str)
      )
      genotype_col = "Genotype_ID"
    else:
      genotype_col = genotype_col_choice
      df[genotype_col] = df[genotype_col].astype(str)

    # Trait and Marker Column Separation
    ignore_cols = [
        line_col,
        tester_col,
        env_col,
        rep_col,
        genotype_col,
        "Genotype_ID",
    ]
    available_traits = [
        c
        for c in df.columns
        if c not in ignore_cols and not str(c).startswith("M_")
    ]
    marker_cols = [c for c in df.columns if str(c).startswith("M_")]

    if not available_traits:
      st.error(
          "❌ No quantitative trait columns found. Check your file mapping."
      )
      st.stop()

    default_traits = [c for c in available_traits if "yield" in str(c).lower()] or [
        available_traits[0]
    ]
    trait_cols = st.sidebar.multiselect(
        "Target Trait(s):", available_traits, default=default_traits
    )

    if not trait_cols:
      st.warning("⚠️ Select at least one quantitative trait to analyze.")
      st.stop()

    primary_trait = st.selectbox("🎯 Focus Trait for Prediction:", trait_cols)

    # Numeric coercion for traits
    for t in trait_cols:
      df[t] = pd.to_numeric(df[t], errors="coerce")

    clean_df = df.dropna(
        subset=[env_col, genotype_col, primary_trait]
    ).copy()

    if clean_df.empty:
      st.error("❌ Dataset is empty after dropping missing values.")
      st.stop()

    # --- Metrics Dashboard Overview ---
    st.markdown("### 📊 Dataset Overview")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Observations", len(clean_df))
    m2.metric("Unique Genotypes", clean_df[genotype_col].nunique())
    m3.metric("Female Lines", clean_df[line_col].nunique())
    m4.metric("Male Testers", clean_df[tester_col].nunique())
    m5.metric(f"Mean {primary_trait}", f"{clean_df[primary_trait].mean():.2f}")

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📋 Data Inspector & Setup",
        "🧬 Parental Breeding Values (EBV/GCA)",
        "🔮 In-Silico Cross Prediction (P1 × P2)",
        "🌳 Heterotic Diversity & Clusters",
        "🌍 GxE Finlay-Wilkinson Stability",
        "🕸️ Multi-Trait Selection Index",
        "📥 Export Reports",
    ])

    # ====================================================
    # TAB 1: DATA INSPECTOR & SETUP
    # ====================================================
    with tab1:
      st.subheader("📋 Dataset Inspection & Structure Guide")
      c_guide, c_prev = st.columns([1, 1])

      with c_guide:
        st.markdown("""
                #### Expected Format Checklist:
                * **Female Line:** Identifier for female parent (e.g. `Line_01`).
                * **Male Tester:** Identifier for male parent (e.g. `Tester_01`).
                * **Location/Env:** Trial environments (e.g. `Env_North`, `Env_Dry`).
                * **Replication:** Block or Rep number (`1`, `2`).
                * **Traits:** Yield, Height, Flowering time, Quality parameters.
                * *(Optional) Markers:* Prefix SNP columns with `M_` (e.g. `M_Marker1`, values: 0, 1, 2) for Genomic Prediction.
                """)

      with c_prev:
        st.write("#### Uploaded Data Preview (First 10 Rows)")
        st.dataframe(clean_df.head(10), use_container_width=True)

      st.write("#### Quantitative Trait Summary Statistics")
      st.dataframe(clean_df[trait_cols].describe().T, use_container_width=True)

    # ====================================================
    # TAB 2: PARENTAL BREEDING VALUES (EBV & GCA/SCA)
    # ====================================================
    with tab2:
      st.subheader(f"Parental Breeding Value & Combining Ability: {primary_trait}")

      overall_trait_mean = clean_df[primary_trait].mean()

      # Female Lines GCA & EBV
      female_means = clean_df.groupby(line_col)[primary_trait].mean()
      gca_female = (female_means - overall_trait_mean).reset_index()
      gca_female.columns = [line_col, "GCA_Female"]
      gca_female["EBV_Female"] = (
          overall_trait_mean + 2 * gca_female["GCA_Female"]
      )
      gca_female["Status"] = np.where(
          gca_female["GCA_Female"] >= 0, "Positive (+)", "Negative (-)"
      )

      # Male Testers GCA & EBV
      male_means = clean_df.groupby(tester_col)[primary_trait].mean()
      gca_male = (male_means - overall_trait_mean).reset_index()
      gca_male.columns = [tester_col, "GCA_Male"]
      gca_male["EBV_Male"] = overall_trait_mean + 2 * gca_male["GCA_Male"]
      gca_male["Status"] = np.where(
          gca_male["GCA_Male"] >= 0, "Positive (+)", "Negative (-)"
      )

      # Cross SCA
      cross_df = (
          clean_df.groupby([line_col, tester_col, genotype_col])[primary_trait]
          .mean()
          .reset_index()
      )
      cross_df = cross_df.merge(gca_female, on=line_col).merge(
          gca_male, on=tester_col
      )
      cross_df["SCA_Cross"] = (
          cross_df[primary_trait]
          - overall_trait_mean
          - cross_df["GCA_Female"]
          - cross_df["GCA_Male"]
      )

      # GEBV genomic modeling if markers available
      has_gebv = False
      parent_gebv_dict = {}

      if marker_cols and len(clean_df) >= 10:
        try:
          X = clean_df[marker_cols].fillna(0)
          y = clean_df[primary_trait]
          if X.var().sum() > 0:
            ridge = Ridge(alpha=1.0)
            ridge.fit(X, y)
            clean_df["Predicted_GEBV"] = ridge.predict(X)
            has_gebv = True

            # Extract parent average marker GEBV
            parent_marker_df = (
                clean_df.groupby(line_col)[marker_cols].mean().fillna(0)
            )
            parent_gebvs = ridge.predict(parent_marker_df)
            for p_name, g_val in zip(parent_marker_df.index, parent_gebvs):
              parent_gebv_dict[str(p_name)] = g_val
        except Exception as e:
          st.warning(f"Genomic model note: {e}")

      c1, c2 = st.columns(2)

      with c1:
        st.write("#### Female Lines General Combining Ability (GCA)")
        fig_gcal = px.bar(
            gca_female.sort_values("GCA_Female"),
            x="GCA_Female",
            y=line_col,
            orientation="h",
            color="Status",
            color_discrete_map={
                "Positive (+)": "#2ca02c",
                "Negative (-)": "#d62728",
            },
            title="Female Parent GCA (Additive Effect)",
            text_auto=".2f",
        )
        fig_gcal.add_vline(x=0, line_dash="dash", line_color="black")
        st.plotly_chart(fig_gcal, use_container_width=True)

      with c2:
        st.write("#### Male Testers General Combining Ability (GCA)")
        fig_gcat = px.bar(
            gca_male.sort_values("GCA_Male"),
            x="GCA_Male",
            y=tester_col,
            orientation="h",
            color="Status",
            color_discrete_map={
                "Positive (+)": "#1f77b4",
                "Negative (-)": "#ff7f0e",
            },
            title="Male Parent GCA (Additive Effect)",
            text_auto=".2f",
        )
        fig_gcat.add_vline(x=0, line_dash="dash", line_color="black")
        st.plotly_chart(fig_gcat, use_container_width=True)

      st.write("#### Specific Combining Ability (SCA) Matrix Heatmap")
      # FIX: Replaced .pivot with .pivot_table to safely handle duplicate entries
      sca_pivot = cross_df.pivot_table(
          index=line_col,
          columns=tester_col,
          values="SCA_Cross",
          aggfunc="mean",
      )
      fig_sca_hm = px.imshow(
          sca_pivot,
          text_auto=".2f",
          color_continuous_scale="RdBu_r",
          title="SCA Heatmap (Line × Tester Specific Hybrid Non-Additive Effects)",
          aspect="auto",
      )
      st.plotly_chart(fig_sca_hm, use_container_width=True)

      st.write("#### 📄 Complete Combining Ability & EBV Table")
      st.dataframe(
          cross_df[[
              genotype_col,
              line_col,
              tester_col,
              primary_trait,
              "GCA_Female",
              "GCA_Male",
              "SCA_Cross",
          ]].sort_values(by="SCA_Cross", ascending=False),
          use_container_width=True,
      )

    # ====================================================
    # TAB 3: IN-SILICO CROSS PREDICTION
    # ====================================================
    with tab3:
      st.subheader("🔮 In-Silico Cross Prediction Engine")
      st.markdown(
          "Predict performance, complementary diversity, and expected"
          " offspring value for **all possible future parental crosses** ($P_1"           " \\times P_2$) before executing them in the field."
      )

      # Unique list of candidate parents (string-cast)
      all_parents = sorted(
          list(
              set(clean_df[line_col].astype(str)).union(
                  set(clean_df[tester_col].astype(str))
              )
          )
      )

      # Build parental mean trait profile with duplicate index resolution
      line_trait_profile = clean_df.groupby(line_col)[trait_cols].mean()
      tester_trait_profile = clean_df.groupby(tester_col)[trait_cols].mean()
      parent_profile = (
          pd.concat([line_trait_profile, tester_trait_profile])
          .groupby(level=0)
          .mean()
      )

      # Calculate parental breeding values
      parent_ebvs = {}
      for p in all_parents:
        if p in gca_female[line_col].values:
          parent_ebvs[p] = gca_female.loc[
              gca_female[line_col] == p, "EBV_Female"
          ].values[0]
        elif p in gca_male[tester_col].values:
          parent_ebvs[p] = gca_male.loc[
              gca_male[tester_col] == p, "EBV_Male"
          ].values[0]
        else:
          parent_ebvs[p] = overall_trait_mean

      # Pairwise combinations
      possible_crosses = list(itertools.combinations(all_parents, 2))
      cross_pred_data = []

      # Compute pairwise distances for genetic divergence
      parent_dist_matrix = squareform(
          pdist(parent_profile.fillna(0), metric="euclidean")
      )
      parent_dist_df = pd.DataFrame(
          parent_dist_matrix,
          index=parent_profile.index,
          columns=parent_profile.index,
      )

      for p1, p2 in possible_crosses:
        ebv1 = parent_ebvs.get(p1, overall_trait_mean)
        ebv2 = parent_ebvs.get(p2, overall_trait_mean)

        mid_parent_ebv = (ebv1 + ebv2) / 2.0

        # Divergence distance
        if p1 in parent_dist_df.index and p2 in parent_dist_df.columns:
          gen_dist = parent_dist_df.loc[p1, p2]
          if isinstance(gen_dist, pd.Series):
            gen_dist = gen_dist.iloc[0]
        else:
          gen_dist = 0.0

        # GEBV prediction if markers available
        if has_gebv and p1 in parent_gebv_dict and p2 in parent_gebv_dict:
          predicted_cross_gebv = (
              parent_gebv_dict[p1] + parent_gebv_dict[p2]
          ) / 2.0
        else:
          predicted_cross_gebv = mid_parent_ebv

        # Check if cross was already tested in dataset
        tested_match = clean_df[
            (
                (clean_df[line_col].astype(str) == p1)
                & (clean_df[tester_col].astype(str) == p2)
            )
            | (
                (clean_df[line_col].astype(str) == p2)
                & (clean_df[tester_col].astype(str) == p1)
            )
        ]
        status = (
            "Already Tested"
            if not tested_match.empty
            else "Untested Hybrid Prediction"
        )

        cross_pred_data.append({
            "Predicted_Cross": f"{p1} × {p2}",
            "Parent_1": p1,
            "Parent_2": p2,
            "Mid_Parent_EBV": mid_parent_ebv,
            "Predicted_GEBV": predicted_cross_gebv,
            "Parental_Divergence_Distance": gen_dist,
            "Status": status,
        })

      pred_cross_df = pd.DataFrame(cross_pred_data).sort_values(
          by="Predicted_GEBV", ascending=False
      )

      col_px1, col_px2 = st.columns(2)

      with col_px1:
        st.write("#### Top 15 Recommended In-Silico Crosses")
        fig_top_cross = px.bar(
            pred_cross_df.head(15),
            x="Predicted_GEBV",
            y="Predicted_Cross",
            orientation="h",
            color="Parental_Divergence_Distance",
            color_continuous_scale="Viridis",
            title="Predicted Cross GEBV (Color = Parental Genetic Divergence)",
            text_auto=".2f",
        )
        fig_top_cross.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_top_cross, use_container_width=True)

      with col_px2:
        st.write("#### Predicted Yield vs Parental Divergence (Heterosis Potential)")
        fig_cross_scat = px.scatter(
            pred_cross_df,
            x="Parental_Divergence_Distance",
            y="Predicted_GEBV",
            color="Status",
            text="Predicted_Cross",
            title="Mid-Parent Performance vs Genetic Distance",
            labels={
                "Parental_Divergence_Distance": "Parental Diversity Distance",
                "Predicted_GEBV": "Predicted Cross Yield / GEBV",
            },
        )
        fig_cross_scat.update_traces(
            textposition="top center", marker=dict(size=10)
        )
        st.plotly_chart(fig_cross_scat, use_container_width=True)

      st.write("#### 📄 Complete Predictive Parental Cross Ranking Table")
      st.dataframe(pred_cross_df, use_container_width=True)

    # ====================================================
    # TAB 4: HETEROTIC DIVERSITY & CLUSTERING
    # ====================================================
    with tab4:
      st.subheader("🌳 Genotypic Diversity & Heterotic Grouping")
      st.markdown(
          "Identify distinct genetic diversity clusters to prevent inbreeding"
          " and structure heterotic pools."
      )

      if marker_cols:
        div_source = st.radio(
            "Diversity Matrix Source:",
            ["Molecular Markers (M_)", "Phenotypic Trait Profiles"],
            horizontal=True,
        )
      else:
        div_source = "Phenotypic Trait Profiles"

      if div_source == "Molecular Markers (M_)":
        div_feat_df = (
            clean_df.groupby(genotype_col)[marker_cols].mean().fillna(0)
        )
      else:
        div_feat_df = (
            clean_df.groupby(genotype_col)[trait_cols]
            .mean()
            .fillna(clean_df[trait_cols].mean())
        )

      if len(div_feat_df) >= 2:
        max_k = min(6, len(div_feat_df))
        k_clusters = st.slider(
            "Select Number of Diversity Clusters (K):",
            2,
            max_k if max_k > 2 else 2,
            min(3, max_k),
        )

        n_comp = min(2, div_feat_df.shape[1])
        pca = PCA(n_components=n_comp)
        coords = pca.fit_transform(div_feat_df)

        kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(div_feat_df)

        cluster_df = pd.DataFrame({
            genotype_col: div_feat_df.index,
            "PC1": coords[:, 0],
            "PC2": coords[:, 1] if n_comp > 1 else np.zeros(len(coords)),
            "Diversity_Cluster": [f"Cluster_{c+1}" for c in clusters],
        })

        col_d1, col_d2 = st.columns(2)

        with col_d1:
          st.write("#### 2D Population Structure (PCA Space)")
          fig_pca = px.scatter(
              cluster_df,
              x="PC1",
              y="PC2",
              color="Diversity_Cluster",
              text=genotype_col,
              title="2D Principal Component Diversity Plot",
              color_discrete_sequence=px.colors.qualitative.Bold,
          )
          fig_pca.update_traces(
              textposition="top center", marker=dict(size=12)
          )
          st.plotly_chart(fig_pca, use_container_width=True)

        with col_d2:
          st.write("#### Pairwise Genetic Distance Matrix Heatmap")
          dist_m = squareform(pdist(div_feat_df, metric="euclidean"))
          dist_df = pd.DataFrame(
              dist_m, index=div_feat_df.index, columns=div_feat_df.index
          )

          fig_hm = px.imshow(
              dist_df,
              color_continuous_scale="Viridis_r",
              title="Pairwise Distance Heatmap (Yellow = Higher Diversity)",
              aspect="auto",
          )
          st.plotly_chart(fig_hm, use_container_width=True)

        st.write("#### 📄 Cluster Assignment Table")
        st.dataframe(
            cluster_df.sort_values("Diversity_Cluster"),
            use_container_width=True,
        )
      else:
        st.warning("⚠️ At least 2 entries are required for diversity clustering.")

    # ====================================================
    # TAB 5: GxE FINLAY-WILKINSON STABILITY
    # ====================================================
    with tab5:
      st.subheader(
          f"🌍 Multi-Environment GxE Stability (Finlay-Wilkinson Model):"
          f" {primary_trait}"
      )
      st.markdown(
          "Joint regression model: **Yield = Mean + $\\beta_i$ (Env_Index)**."
          " Slope $\\beta_i$ measures environmental sensitivity."
      )

      env_means = clean_df.groupby(env_col)[primary_trait].mean()
      grand_mean = clean_df[primary_trait].mean()
      env_index = env_means - grand_mean

      fw_rows = []
      for g in clean_df[genotype_col].unique():
        sub = clean_df[clean_df[genotype_col] == g]
        g_env_means = sub.groupby(env_col)[primary_trait].mean()

        aligned = (
            pd.DataFrame({"Env_Index": env_index, "Geno_Mean": g_env_means})
            .dropna()
        )

        if len(aligned) >= 2 and aligned["Env_Index"].std() > 1e-6:
          slope, intercept, r_val, p_val, std_err = linregress(
              aligned["Env_Index"], aligned["Geno_Mean"]
          )
          mean_p = aligned["Geno_Mean"].mean()

          if slope > 1.1:
            cat = "Favorable Environment Specialist (β > 1.1)"
          elif slope < 0.9:
            cat = "Stress Tolerant / Low-Input Specialist (β < 0.9)"
          else:
            cat = "Broadly Adapted / Stable (β ≈ 1.0)"

          fw_rows.append({
              genotype_col: g,
              "Mean_Performance": mean_p,
              "Regression_Slope_Beta": slope,
              "R2_Stability": r_val**2 if not np.isnan(r_val) else 0.0,
              "Adaptation_Category": cat,
          })

      fw_df = pd.DataFrame(fw_rows)

      if not fw_df.empty:
        c_fw1, c_fw2 = st.columns(2)

        with c_fw1:
          st.write("#### Adaptation Scatter Diagram")
          fig_fw = px.scatter(
              fw_df,
              x="Regression_Slope_Beta",
              y="Mean_Performance",
              color="Adaptation_Category",
              text=genotype_col,
              title="Performance Mean vs Environmental Sensitivity (Slope β)",
              labels={
                  "Regression_Slope_Beta": "Environmental Sensitivity (Slope β)",
                  "Mean_Performance": f"Mean {primary_trait}",
              },
          )
          fig_fw.add_vline(
              x=1.0, line_dash="dash", annotation_text="Average Slope β=1"
          )
          fig_fw.add_hline(
              y=grand_mean, line_dash="dash", annotation_text="Grand Mean"
          )
          fig_fw.update_traces(textposition="top center", marker=dict(size=10))
          st.plotly_chart(fig_fw, use_container_width=True)

        with c_fw2:
          st.write("#### Reaction Norm Plot Across Environments")
          rn_df = (
              clean_df.groupby([genotype_col, env_col])[primary_trait]
              .mean()
              .reset_index()
          )
          fig_rn = px.line(
              rn_df,
              x=env_col,
              y=primary_trait,
              color=genotype_col,
              markers=True,
              title="Reaction Norms Across Locations",
          )
          st.plotly_chart(fig_rn, use_container_width=True)

        st.write("#### 📄 Finlay-Wilkinson Stability Summary")
        st.dataframe(
            fw_df.sort_values(by="Mean_Performance", ascending=False),
            use_container_width=True,
        )
      else:
        st.warning(
            "⚠️ Finlay-Wilkinson regression requires data across at least 2"
            " distinct environments."
        )

    # ====================================================
    # TAB 6: MULTI-TRAIT SELECTION INDEX
    # ====================================================
    with tab6:
      st.subheader("🕸️ Multi-Trait Parental Selection Index & Radar Profile")

      if len(trait_cols) > 1:
        st.write(
            "Assign relative economic weights to multi-trait Z-scores to build a"
            " unified **Selection Index Score**."
        )

        weights = {}
        w_cols = st.columns(len(trait_cols))
        for idx, t in enumerate(trait_cols):
          with w_cols[idx]:
            weights[t] = st.slider(f"Weight: {t}", -5.0, 5.0, 1.0, step=0.5)

        parent_means = clean_df.groupby(genotype_col)[trait_cols].mean()
        std_devs = parent_means.std()
        std_devs[std_devs == 0] = 1.0
        z_scores = (parent_means - parent_means.mean()) / std_devs

        idx_scores = np.zeros(len(parent_means))
        for t in trait_cols:
          idx_scores += z_scores[t] * weights[t]

        parent_means["Selection_Index_Score"] = idx_scores
        parent_means = parent_means.sort_values(
            by="Selection_Index_Score", ascending=False
        )

        cr1, cr2 = st.columns(2)

        with cr1:
          st.write("#### Top Ranked Genotypes by Selection Index")
          fig_idx = px.bar(
              parent_means.head(10).reset_index(),
              x="Selection_Index_Score",
              y=genotype_col,
              orientation="h",
              color="Selection_Index_Score",
              color_continuous_scale="Viridis",
              title="Top 10 Multi-Trait Selection Index",
              text_auto=".2f",
          )
          fig_idx.update_layout(yaxis={"categoryorder": "total ascending"})
          st.plotly_chart(fig_idx, use_container_width=True)

        with cr2:
          st.write("#### Multi-Trait Radar Profiles (Top 5 Genotypes)")
          top5 = parent_means.head(5).index
          min_v = parent_means[trait_cols].min()
          max_v = parent_means[trait_cols].max()
          diff_v = max_v - min_v
          diff_v[diff_v == 0] = 1.0
          norm_radar = (parent_means[trait_cols] - min_v) / diff_v

          fig_radar = go.Figure()
          for g in top5:
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=norm_radar.loc[g].values,
                    theta=trait_cols,
                    fill="toself",
                    name=str(g),
                )
            )
          fig_radar.update_layout(
              polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
              title="Normalized Trait Profile Comparison",
          )
          st.plotly_chart(fig_radar, use_container_width=True)

        st.write("#### 📄 Multi-Trait Index Table")
        st.dataframe(parent_means, use_container_width=True)
      else:
        st.info(
            "💡 Select multiple traits in the sidebar to activate the"
            " Multi-Trait Selection Index."
        )

    # ====================================================
    # TAB 7: EXPORT MASTER REPORTS
    # ====================================================
    with tab7:
      st.subheader("📥 Export Complete Predictive Breeding Report")
      st.markdown(
          "Download a multi-sheet Excel workbook containing raw data,"
          " cross predictions, combining ability stats, stability parameters,"
          " and cluster assignments."
      )

      master_buffer = io.BytesIO()
      with pd.ExcelWriter(master_buffer, engine="xlsxwriter") as writer:
        clean_df.to_excel(writer, sheet_name="Trial_Data", index=False)
        if "pred_cross_df" in locals():
          pred_cross_df.to_excel(
              writer, sheet_name="InSilico_Cross_Predictions", index=False
          )
        if "gca_female" in locals():
          gca_female.to_excel(
              writer, sheet_name="Female_GCA_EBV", index=False
          )
          gca_male.to_excel(writer, sheet_name="Male_GCA_EBV", index=False)
          cross_df.to_excel(writer, sheet_name="SCA_Crosses", index=False)
        if "cluster_df" in locals():
          cluster_df.to_excel(
              writer, sheet_name="Diversity_Clusters", index=False
          )
        if "fw_df" in locals():
          fw_df.to_excel(writer, sheet_name="GxE_Stability_FW", index=False)
        if "parent_means" in locals():
          parent_means.to_excel(
              writer, sheet_name="MultiTrait_Selection_Index", index=True
          )

      st.download_button(
          label="📥 Download Master Predictive Breeding Report (.xlsx)",
          data=master_buffer.getvalue(),
          file_name="Master_Predictive_Breeding_Report.xlsx",
          mime="application/vnd.ms-excel",
      )

  except Exception as err:
    st.error(f"❌ Error executing predictive breeding analysis: {err}")

else:
  st.info(
      "👈 Upload an Excel (.xlsx) or CSV (.csv) trial file in the sidebar to"
      " begin, or click 'Download Predictive Breeding Excel Template' to test"
      " the portal."
  )
