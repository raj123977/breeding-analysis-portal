import io
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
    "Enterprise Decision Support System for **Parental Breeding Value"
    " Estimation (EBV/GEBV)**, **Diversity Clustering**, **GxE"
    " Finlay-Wilkinson Stability**, and **Multi-Environment MET Analysis**."
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
          # Mock marker columns
          m1 = np.random.choice([0, 1, 2])
          m2 = np.random.choice([0, 1, 2])
          m3 = np.random.choice([0, 1, 2])

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

# Download Template Button
template_bytes = generate_sample_template()
st.sidebar.download_button(
    label="📥 Download Sample Excel Template",
    data=template_bytes,
    file_name="predictive_breeding_sample_template.xlsx",
    mime="application/vnd.ms-excel",
    help="Download a correctly formatted sample Excel template to inspect the expected column structure.",
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
    st.sidebar.success("Data loaded successfully!")

    cols = [str(c) for c in df.columns.tolist()]

    # Automatic Index Matching Helper
    def get_default_index(target_names, col_list, fallback=0):
      for name in target_names:
        for idx, col in enumerate(col_list):
          if name.lower() in col.lower():
            return idx
      return fallback

    st.sidebar.header("⚙️ Variable Mapping")

    # 1. Genotype ID Mapping Option
    geno_idx = get_default_index(
        ["genotype", "entry", "hybrid", "variety"], cols, fallback=None
    )
    genotype_options = ["Auto-generate from Female × Male"] + cols

    if geno_idx is not None:
      default_geno_choice = cols[geno_idx]
      selected_geno_idx = genotype_options.index(default_geno_choice)
    else:
      selected_geno_idx = 0

    genotype_col_choice = st.sidebar.selectbox(
        "Genotype ID Column:", genotype_options, index=selected_geno_idx
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

    line_col = st.sidebar.selectbox("Line Column (Female):", cols, index=line_idx)
    tester_col = st.sidebar.selectbox(
        "Tester Column (Male):", cols, index=tester_idx
    )
    env_col = st.sidebar.selectbox(
        "Location / Environment Column:", cols, index=env_idx
    )
    rep_col = st.sidebar.selectbox("Replication Column:", cols, index=rep_idx)

    # Resolve Genotype ID Column
    if genotype_col_choice == "Auto-generate from Female × Male":
      df["Genotype_ID"] = (
          df[line_col].astype(str) + " × " + df[tester_col].astype(str)
      )
      genotype_col = "Genotype_ID"
    else:
      genotype_col = genotype_col_choice

    # Trait & Marker Column Separator
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
          "❌ No numeric trait columns found in dataset. Please check your"
          " file mapping."
      )
      st.stop()

    default_traits = [c for c in available_traits if "yield" in str(c).lower()] or [
        available_traits[0]
    ]

    trait_cols = st.sidebar.multiselect(
        "Target Trait(s):", available_traits, default=default_traits
    )

    if not trait_cols:
      st.warning("⚠️ Select at least one numeric trait column in the sidebar.")
      st.stop()

    primary_trait = st.selectbox("🎯 Primary Focus Trait:", trait_cols)

    # Clean numeric traits
    for t in trait_cols:
      df[t] = pd.to_numeric(df[t], errors="coerce")

    # Clean subset dataframe
    clean_df = df.dropna(
        subset=[env_col, genotype_col, primary_trait]
    ).copy()

    if clean_df.empty:
      st.error(
          "❌ Clean dataset is empty after dropping missing values. Please"
          " verify column selection."
      )
      st.stop()

    # --- Top Dashboard Overview Metrics ---
    st.markdown("### 📊 Dataset Overview")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Rows", len(clean_df))
    m2.metric("Unique Genotypes", clean_df[genotype_col].nunique())
    m3.metric("Female Lines", clean_df[line_col].nunique())
    m4.metric("Male Testers", clean_df[tester_col].nunique())

    mean_val = clean_df[primary_trait].mean()
    m5.metric(f"Mean {primary_trait}", f"{mean_val:.2f}")

    # Navigation Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 Excel Format & Data Inspector",
        "🧬 Breeding Values (EBV/GEBV & GCA/SCA)",
        "🌳 Diversity & Cluster Analysis",
        "🌍 GxE Stability (Finlay-Wilkinson)",
        "🕸️ Multi-Trait Selection Index",
        "📥 Export Reports",
    ])

    # ====================================================
    # TAB 1: EXCEL FORMAT GUIDE & DATA INSPECTOR
    # ====================================================
    with tab1:
      st.subheader("📋 Dataset Inspection & Formatting Guidelines")

      col_guide, col_preview = st.columns([1, 1])

      with col_guide:
        st.markdown("""
                #### Recommended Excel Column Structure:
                To achieve seamless analysis without errors, ensure your dataset includes:
                
                * **Genotype ID Column:** Unique entry code or hybrid name (e.g., `G_01`, `Line_01 × Tester_01`).
                * **Female Line Column:** Name or ID of Female Parent (e.g., `Line_01`, `L_102`).
                * **Male Tester Column:** Name or ID of Male Parent (e.g., `Tester_01`, `T_1`).
                * **Environment / Location:** Trial location code (e.g., `Env_North`, `Loc_A`).
                * **Replication / Block:** Numeric replication number (`1`, `2`, `3`).
                * **Quantitative Trait Columns:** Yield, Days to Flower, Plant Height, etc.
                * *(Optional) Marker Columns:* Prefix molecular SNP columns with `M_` (e.g., `M_SNP001` scored 0, 1, 2) for Genomic Selection (GEBV).
                """)

      with col_preview:
        st.write("#### Uploaded Data Sample (First 10 Rows)")
        st.dataframe(clean_df.head(10), use_container_width=True)

      st.write("#### Quantitative Trait Summary Statistics")
      st.dataframe(clean_df[trait_cols].describe().T, use_container_width=True)

    # ====================================================
    # TAB 2: BREEDING VALUES (EBV/GEBV & GCA/SCA)
    # ====================================================
    with tab2:
      st.subheader(
          f"Breeding Value Estimation & Combining Ability: {primary_trait}"
      )

      lt_data = clean_df.dropna(
          subset=[line_col, tester_col, primary_trait]
      ).copy()
      overall_trait_mean = lt_data[primary_trait].mean()

      # 1. GCA Female Lines
      line_means = lt_data.groupby(line_col)[primary_trait].mean()
      gca_lines = (line_means - overall_trait_mean).reset_index()
      gca_lines.columns = [line_col, "GCA_Female"]
      gca_lines["EBV_Female"] = (
          overall_trait_mean + 2 * gca_lines["GCA_Female"]
      )
      gca_lines["Status"] = np.where(
          gca_lines["GCA_Female"] >= 0, "Positive (+)", "Negative (-)"
      )

      # 2. GCA Male Testers
      tester_means = lt_data.groupby(tester_col)[primary_trait].mean()
      gca_testers = (tester_means - overall_trait_mean).reset_index()
      gca_testers.columns = [tester_col, "GCA_Male"]
      gca_testers["EBV_Male"] = overall_trait_mean + 2 * gca_testers["GCA_Male"]
      gca_testers["Status"] = np.where(
          gca_testers["GCA_Male"] >= 0, "Positive (+)", "Negative (-)"
      )

      # 3. SCA Hybrids/Crosses
      cross_means = (
          lt_data.groupby([line_col, tester_col, genotype_col])[primary_trait]
          .mean()
          .reset_index()
      )
      cross_means = cross_means.merge(gca_lines, on=line_col).merge(
          gca_testers, on=tester_col
      )
      cross_means["SCA_Cross"] = (
          cross_means[primary_trait]
          - overall_trait_mean
          - cross_means["GCA_Female"]
          - cross_means["GCA_Male"]
      )

      # Genomic Selection (GEBV) via Ridge Regression if markers exist
      has_gebv = False
      if marker_cols and len(lt_data) > 5:
        try:
          X = lt_data[marker_cols].fillna(0)
          y = lt_data[primary_trait]
          if X.var().sum() > 0:  # Check non-zero marker variance
            ridge_model = Ridge(alpha=1.0)
            ridge_model.fit(X, y)
            lt_data["Predicted_GEBV"] = ridge_model.predict(X)
            has_gebv = True
        except Exception as e:
          st.warning(f"GEBV calculation bypassed: {e}")

      # VISUALIZATIONS
      c1, c2 = st.columns(2)

      with c1:
        st.write("#### Female Lines GCA Effects (Additive Value)")
        fig_gcal = px.bar(
            gca_lines.sort_values(by="GCA_Female"),
            x="GCA_Female",
            y=line_col,
            orientation="h",
            color="Status",
            color_discrete_map={
                "Positive (+)": "#2ca02c",
                "Negative (-)": "#d62728",
            },
            title="Female Parent GCA (Diverging from Zero)",
            text_auto=".2f",
        )
        fig_gcal.add_vline(x=0, line_dash="dash", line_color="black")
        st.plotly_chart(fig_gcal, use_container_width=True)

      with c2:
        st.write("#### Male Testers GCA Effects (Additive Value)")
        fig_gcat = px.bar(
            gca_testers.sort_values(by="GCA_Male"),
            x="GCA_Male",
            y=tester_col,
            orientation="h",
            color="Status",
            color_discrete_map={
                "Positive (+)": "#1f77b4",
                "Negative (-)": "#ff7f0e",
            },
            title="Male Parent GCA (Diverging from Zero)",
            text_auto=".2f",
        )
        fig_gcat.add_vline(x=0, line_dash="dash", line_color="black")
        st.plotly_chart(fig_gcat, use_container_width=True)

      st.write("#### Specific Combining Ability (SCA) Matrix Heatmap")
      sca_pivot = cross_means.pivot(
          index=line_col, columns=tester_col, values="SCA_Cross"
      )
      fig_sca_hm = px.imshow(
          sca_pivot,
          text_auto=".2f",
          color_continuous_scale="RdBu_r",
          title="SCA Heatmap (Line × Tester Specific Non-Additive Effects)",
          aspect="auto",
      )
      st.plotly_chart(fig_sca_hm, use_container_width=True)

      if has_gebv:
        st.write("#### Genomic Estimated Breeding Values (GEBV) Validation")
        fig_gebv = px.scatter(
            lt_data,
            x=primary_trait,
            y="Predicted_GEBV",
            color=line_col,
            trendline="ols",
            title=f"Observed {primary_trait} vs Genomic Predicted Value (GEBV)",
            labels={
                primary_trait: f"Observed {primary_trait}",
                "Predicted_GEBV": "Predicted GEBV",
            },
        )
        st.plotly_chart(fig_gebv, use_container_width=True)

      st.write("#### 📄 Complete Combining Ability & EBV Table")
      st.dataframe(
          cross_means[[
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
    # TAB 3: DIVERSITY & CLUSTER ANALYSIS
    # ====================================================
    with tab3:
      st.subheader("🌳 Genotypic Diversity & Cluster Analysis")
      st.markdown(
          "Cluster genotypes into distinct genetic groups based on target traits"
          " or molecular markers to optimize heterotic pairings."
      )

      if marker_cols:
        cluster_source = st.radio(
            "Clustering Data Source:",
            ["Molecular Markers (M_)", "Phenotypic Trait Profiles"],
            horizontal=True,
        )
      else:
        cluster_source = "Phenotypic Trait Profiles"

      if cluster_source == "Molecular Markers (M_)":
        feat_df = (
            clean_df.groupby(genotype_col)[marker_cols].mean().fillna(0)
        )
      else:
        feat_df = (
            clean_df.groupby(genotype_col)[trait_cols]
            .mean()
            .fillna(clean_df[trait_cols].mean())
        )

      if len(feat_df) >= 2:
        max_k = min(6, len(feat_df))
        n_clusters = st.slider(
            "Select Number of Genetic Clusters (K):",
            2,
            max_k if max_k > 2 else 2,
            min(3, max_k),
        )

        # 2D PCA Transformation
        n_comp = min(2, feat_df.shape[1])
        pca = PCA(n_components=n_comp)
        coords_2d = pca.fit_transform(feat_df)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(feat_df)

        pca_df = pd.DataFrame({
            genotype_col: feat_df.index,
            "PC1": coords_2d[:, 0],
            "PC2": (
                coords_2d[:, 1]
                if n_comp > 1
                else np.zeros(len(coords_2d))
            ),
            "Cluster": [f"Cluster {c+1}" for c in cluster_labels],
        })

        col_div1, col_div2 = st.columns(2)

        with col_div1:
          st.write("#### 2D Genetic Diversity PCA Plot")
          fig_pca_2d = px.scatter(
              pca_df,
              x="PC1",
              y="PC2",
              color="Cluster",
              text=genotype_col,
              title="2D Principal Component Space",
              color_discrete_sequence=px.colors.qualitative.Set1,
          )
          fig_pca_2d.update_traces(
              textposition="top center", marker=dict(size=12)
          )
          st.plotly_chart(fig_pca_2d, use_container_width=True)

        with col_div2:
          st.write("#### Pairwise Distance Heatmap")
          dist_matrix = squareform(pdist(feat_df, metric="euclidean"))
          dist_df = pd.DataFrame(
              dist_matrix, index=feat_df.index, columns=feat_df.index
          )

          fig_dist = px.imshow(
              dist_df,
              color_continuous_scale="Viridis_r",
              title="Pairwise Distance Heatmap (Yellow = Greater Diversity)",
              aspect="auto",
          )
          st.plotly_chart(fig_dist, use_container_width=True)

        st.write("#### 📄 Cluster Assignment Table")
        st.dataframe(
            pca_df.sort_values(by="Cluster"), use_container_width=True
        )
      else:
        st.warning("⚠️ At least 2 genotypes are required for cluster analysis.")

    # ====================================================
    # TAB 4: GxE STABILITY (FINLAY-WILKINSON REGRESSION)
    # ====================================================
    with tab4:
      st.subheader(
          f"🌍 Genotype × Environment Stability & Adaptation: {primary_trait}"
      )
      st.markdown(
          "**Finlay-Wilkinson Joint Regression Model:** Evaluates parental"
          " mean yield against environmental sensitivity (regression slope"
          " $\\beta_i$)."
      )

      # Environmental Index = Location Mean - Grand Mean
      env_means = clean_df.groupby(env_col)[primary_trait].mean()
      grand_overall_mean = clean_df[primary_trait].mean()
      env_index = env_means - grand_overall_mean

      fw_results = []
      genotypes_list = clean_df[genotype_col].unique()

      for g in genotypes_list:
        sub = clean_df[clean_df[genotype_col] == g]
        geno_env_means = sub.groupby(env_col)[primary_trait].mean()

        aligned = (
            pd.DataFrame({"Env_Index": env_index, "Geno_Mean": geno_env_means})
            .dropna()
        )

        # Regression Guard: Requires >= 2 environments and non-zero x variance
        if len(aligned) >= 2 and aligned["Env_Index"].std() > 1e-6:
          slope, intercept, r_val, p_val, std_err = linregress(
              aligned["Env_Index"], aligned["Geno_Mean"]
          )
          mean_perf = aligned["Geno_Mean"].mean()

          if slope > 1.1:
            adapt = "Favorable Environment Specialist (High Sensitivity)"
          elif slope < 0.9:
            adapt = "Stress Tolerant / Low-Input Specialist (Robust)"
          else:
            adapt = "Broadly Adapted (Stable Across Envs)"

          fw_results.append({
              genotype_col: g,
              "Mean_Performance": mean_perf,
              "Regression_Slope_Beta": slope,
              "R2_Fit": r_val**2 if not np.isnan(r_val) else 0.0,
              "Adaptation_Category": adapt,
          })

      fw_df = pd.DataFrame(fw_results)

      if not fw_df.empty:
        c_fw1, c_fw2 = st.columns(2)

        with c_fw1:
          st.write("#### Finlay-Wilkinson Adaptation Scatter Chart")
          fig_fw_scatter = px.scatter(
              fw_df,
              x="Regression_Slope_Beta",
              y="Mean_Performance",
              color="Adaptation_Category",
              text=genotype_col,
              title="Yield Potential vs Environmental Sensitivity (Slope β)",
              labels={
                  "Regression_Slope_Beta": "Environmental Sensitivity (Slope β)",
                  "Mean_Performance": f"Mean {primary_trait}",
              },
          )
          fig_fw_scatter.add_vline(
              x=1.0, line_dash="dash", annotation_text="Average Sensitivity β=1"
          )
          fig_fw_scatter.add_hline(
              y=grand_overall_mean,
              line_dash="dash",
              annotation_text="Overall Mean",
          )
          fig_fw_scatter.update_traces(
              textposition="top center", marker=dict(size=10)
          )
          st.plotly_chart(fig_fw_scatter, use_container_width=True)

        with c_fw2:
          st.write("#### Reaction Norm Plot Across Trial Locations")
          gxe_line_df = (
              clean_df.groupby([genotype_col, env_col])[primary_trait]
              .mean()
              .reset_index()
          )
          fig_rn = px.line(
              gxe_line_df,
              x=env_col,
              y=primary_trait,
              color=genotype_col,
              markers=True,
              title="Reaction Norms Across Environments",
          )
          st.plotly_chart(fig_rn, use_container_width=True)

        st.write("#### 📄 Finlay-Wilkinson Stability Table")
        st.dataframe(
            fw_df.sort_values(by="Mean_Performance", ascending=False),
            use_container_width=True,
        )
      else:
        st.warning(
            "⚠️ Finlay-Wilkinson model requires genotypes to be evaluated"
            " across at least 2 distinct environments."
        )

    # ====================================================
    # TAB 5: MULTI-TRAIT SELECTION INDEX
    # ====================================================
    with tab5:
      st.subheader("🕸️ Multi-Trait Parental Selection Index & Radar Profile")

      if len(trait_cols) > 1:
        st.write(
            "Assign economic weights to target traits to build a consolidated"
            " **Parental Selection Index**."
        )

        weights = {}
        w_cols = st.columns(len(trait_cols))
        for idx, t in enumerate(trait_cols):
          with w_cols[idx]:
            weights[t] = st.slider(f"Weight for {t}:", -5.0, 5.0, 1.0, step=0.5)

        parent_means = clean_df.groupby(genotype_col)[trait_cols].mean()

        # Z-score standardization
        std_devs = parent_means.std()
        std_devs[std_devs == 0] = 1.0
        z_scores = (parent_means - parent_means.mean()) / std_devs

        index_scores = np.zeros(len(parent_means))
        for t in trait_cols:
          index_scores += z_scores[t] * weights[t]

        parent_means["Selection_Index_Score"] = index_scores
        parent_means = parent_means.sort_values(
            by="Selection_Index_Score", ascending=False
        )

        col_r1, col_r2 = st.columns(2)

        with col_r1:
          st.write("#### Top Ranked Genotypes by Selection Index")
          fig_index_bar = px.bar(
              parent_means.head(10).reset_index(),
              x="Selection_Index_Score",
              y=genotype_col,
              orientation="h",
              color="Selection_Index_Score",
              color_continuous_scale="Viridis",
              title="Top 10 Genotypes Ranked by Multi-Trait Index",
              text_auto=".2f",
          )
          fig_index_bar.update_layout(
              yaxis={"categoryorder": "total ascending"}
          )
          st.plotly_chart(fig_index_bar, use_container_width=True)

        with col_r2:
          st.write("#### Multi-Trait Radar Profile (Top 5 Genotypes)")
          top_5_geno = parent_means.head(5).index
          min_val = parent_means[trait_cols].min()
          max_val = parent_means[trait_cols].max()
          diff = max_val - min_val
          diff[diff == 0] = 1.0

          radar_norm = (parent_means[trait_cols] - min_val) / diff

          fig_radar = go.Figure()
          for g in top_5_geno:
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=radar_norm.loc[g].values,
                    theta=trait_cols,
                    fill="toself",
                    name=str(g),
                )
            )
          fig_radar.update_layout(
              polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
              title="Normalized Trait Comparison Radar Plot",
          )
          st.plotly_chart(fig_radar, use_container_width=True)

        st.write("#### 📄 Multi-Trait Selection Index Table")
        st.dataframe(parent_means, use_container_width=True)
      else:
        st.info(
            "💡 Please select multiple target traits in the sidebar to activate"
            " the Multi-Trait Selection Index."
        )

    # ====================================================
    # TAB 6: EXPORT COMPREHENSIVE EXCEL REPORTS
    # ====================================================
    with tab6:
      st.subheader("📥 Export Master Predictive Breeding Report")
      st.markdown(
          "Download a complete multi-sheet Excel workbook containing all"
          " statistical computations, breeding values, stability metrics, and"
          " cluster assignments."
      )

      master_buffer = io.BytesIO()
      with pd.ExcelWriter(master_buffer, engine="xlsxwriter") as writer:
        clean_df.to_excel(writer, sheet_name="Raw_Trial_Data", index=False)
        if "gca_lines" in locals():
          gca_lines.to_excel(
              writer, sheet_name="Female_Line_GCA_EBV", index=False
          )
          gca_testers.to_excel(
              writer, sheet_name="Male_Tester_GCA_EBV", index=False
          )
          cross_means.to_excel(writer, sheet_name="SCA_Crosses", index=False)
        if "pca_df" in locals():
          pca_df.to_excel(writer, sheet_name="Genetic_Clusters", index=False)
        if "fw_df" in locals():
          fw_df.to_excel(writer, sheet_name="GxE_Stability_FW", index=False)
        if "parent_means" in locals():
          parent_means.to_excel(
              writer, sheet_name="Selection_Index", index=True
          )

      st.download_button(
          label=(
              "📥 Download Complete Multi-Sheet Master Breeding Report (.xlsx)"
          ),
          data=master_buffer.getvalue(),
          file_name="Predictive_Breeding_Master_Report.xlsx",
          mime="application/vnd.ms-excel",
      )

  except Exception as err:
    st.error(f"❌ An error occurred during data processing: {err}")

else:
  st.info(
      "👈 Upload an Excel (.xlsx) or CSV (.csv) trial file in the sidebar to"
      " begin, or download the Sample Template to inspect the expected format."
  )
