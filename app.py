import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import f, linregress
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
    " Estimation (EBV/GEBV)**, **Diversity Clustering**, **GxE Finlay-Wilkinson"
    " Stability**, and **Multi-Environment MET ANOVA**."
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
              "Location": env,
              "Replication": rep,
              "Female_Line": line,
              "Male_Tester": tester,
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

    cols = df.columns.tolist()

    # Automatic Index Matching for Column Defaults
    def get_default_index(target_names, col_list, fallback=0):
      for name in target_names:
        for idx, col in enumerate(col_list):
          if name.lower() in col.lower():
            return idx
      return fallback

    line_idx = get_default_index(
        ["line", "female", "parent1", "mother"], cols, 0
    )
    tester_idx = get_default_index(
        ["tester", "male", "parent2", "father"], cols, min(1, len(cols) - 1)
    )
    env_idx = get_default_index(
        ["location", "loc", "env", "site"], cols, min(2, len(cols) - 1)
    )
    rep_idx = get_default_index(
        ["replication", "rep", "block"], cols, min(3, len(cols) - 1)
    )

    st.sidebar.header("⚙️ Variable Mapping")
    line_col = st.sidebar.selectbox("Line Column (Female):", cols, index=line_idx)
    tester_col = st.sidebar.selectbox(
        "Tester Column (Male):", cols, index=tester_idx
    )
    env_col = st.sidebar.selectbox(
        "Location / Environment Column:", cols, index=env_idx
    )
    rep_col = st.sidebar.selectbox("Replication Column:", cols, index=rep_idx)

    # Generate Genotype Cross Identifier
    df["Genotype_ID"] = (
        df[line_col].astype(str) + " × " + df[tester_col].astype(str)
    )
    genotype_col = "Genotype_ID"

    # Separate Trait columns from Marker columns
    available_traits = [
        c
        for c in cols
        if c not in [line_col, tester_col, env_col, rep_col, genotype_col]
        and not c.startswith("M_")
    ]
    default_traits = [c for c in available_traits if "yield" in c.lower()] or (
        [available_traits[0]] if available_traits else []
    )

    trait_cols = st.sidebar.multiselect(
        "Target Trait(s):", available_traits, default=default_traits
    )
    marker_cols = [c for c in cols if c.startswith("M_")]

    if not trait_cols:
      st.warning("⚠️ Select at least one numeric trait column in the sidebar.")
      st.stop()

    primary_trait = st.selectbox("🎯 Primary Focus Trait:", trait_cols)
    df[primary_trait] = pd.to_numeric(df[primary_trait], errors="coerce")

    # Clean subset
    clean_df = df.dropna(
        subset=[env_col, genotype_col, rep_col, primary_trait]
    ).copy()

    # --- Top Dashboard Metrics ---
    st.markdown("### 📊 Dataset Overview")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Rows", len(clean_df))
    m2.metric("Female Lines", clean_df[line_col].nunique())
    m3.metric("Male Testers", clean_df[tester_col].nunique())
    m4.metric("Environments", clean_df[env_col].nunique())

    # Calculate overall phenotype mean
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
                To achieve seamless analysis without errors, structure your input Excel file with the following columns:
                
                * **Female Line Column:** Name or ID of Female Parent (e.g., `Line_01`, `L_102`)
                * **Male Tester Column:** Name or ID of Male Parent (e.g., `Tester_01`, `T_1`)
                * **Environment / Location:** Trial location code (e.g., `Env_North`, `Loc_A`)
                * **Replication / Block:** Numeric replication number (`1`, `2`, `3`)
                * **Quantitative Trait Columns:** Yield, Days to Flower, Plant Height, etc.
                * *(Optional) SNP Marker Columns:* Prefix marker columns with `M_` (e.g., `M_Marker1`, `M_SNP002` scored as 0, 1, 2) for Genomic Selection (GEBV).
                """)

      with col_preview:
        st.write("#### Uploaded Data Sample (First 10 Rows)")
        st.dataframe(clean_df.head(10), use_container_width=True)

      st.write("#### Data Summary Statistics")
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
          lt_data.groupby([line_col, tester_col])[primary_trait]
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
      cross_means["Cross_Name"] = (
          cross_means[line_col].astype(str)
          + " × "
          + cross_means[tester_col].astype(str)
      )

      # Genomic Selection (GEBV) via Ridge Regression if markers exist
      has_gebv = False
      if marker_cols:
        try:
          X = lt_data[marker_cols].fillna(0)
          y = lt_data[primary_trait]
          ridge_model = Ridge(alpha=1.0)
          ridge_model.fit(X, y)
          lt_data["Predicted_GEBV"] = ridge_model.predict(X)
          has_gebv = True
        except Exception as e:
          st.error(f"Could not calculate GEBV: {e}")

      # VISUALIZATIONS (2D ONLY)
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

      st.write("#### Hybrid Specific Combining Ability (SCA) Matrix Heatmap")
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

      # TABULAR DATA
      st.write("#### 📄 Complete Parental & Cross Combining Ability Table")
      st.dataframe(
          cross_means[[
              "Cross_Name",
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
          "Group parents into genetic clusters based on performance profiles"
          " or molecular markers to avoid crossing closely related parents."
      )

      # Prepare matrix for clustering (either traits or markers)
      if marker_cols:
        cluster_source = st.radio(
            "Clustering Based On:",
            ["Molecular Markers (M_)", "Phenotypic Trait Profiles"],
            horizontal=True,
        )
      else:
        cluster_source = "Phenotypic Trait Profiles"

      if cluster_source == "Molecular Markers (M_)":
        feat_df = clean_df.groupby(line_col)[marker_cols].mean().fillna(0)
      else:
        feat_df = (
            clean_df.groupby(line_col)[trait_cols]
            .mean()
            .fillna(clean_df[trait_cols].mean())
        )

      # K-Means Number of Clusters
      n_clusters = st.slider("Select Number of Genetic Clusters (K):", 2, 6, 3)

      # PCA for 2D Visualization
      pca = PCA(n_components=2)
      coords_2d = pca.fit_transform(feat_df)

      kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
      cluster_labels = kmeans.fit_predict(feat_df)

      pca_df = pd.DataFrame({
          line_col: feat_df.index,
          "PC1": coords_2d[:, 0],
          "PC2": coords_2d[:, 1],
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
            text=line_col,
            title="2D Principal Component Diversity Space",
            color_discrete_sequence=px.colors.qualitative.Set1,
        )
        fig_pca_2d.update_traces(
            textposition="top center", marker=dict(size=12)
        )
        st.plotly_chart(fig_pca_2d, use_container_width=True)

      with col_div2:
        st.write("#### Pairwise Euclidean Distance Matrix Heatmap")
        dist_matrix = squareform(pdist(feat_df, metric="euclidean"))
        dist_df = pd.DataFrame(
            dist_matrix, index=feat_df.index, columns=feat_df.index
        )

        fig_dist = px.imshow(
            dist_df,
            color_continuous_scale="Viridis_r",
            title="Pairwise Distance Heatmap (Yellow = High Diversity)",
            aspect="auto",
        )
        st.plotly_chart(fig_dist, use_container_width=True)

      st.write("#### 📄 Cluster Assignment Table")
      st.dataframe(pca_df.sort_values(by="Cluster"), use_container_width=True)

    # ====================================================
    # TAB 4: GxE STABILITY (FINLAY-WILKINSON REGRESSION)
    # ====================================================
    with tab4:
      st.subheader(
          f"🌍 Genotype × Environment Stability & Adaptation: {primary_trait}"
      )
      st.markdown(
          "**Finlay-Wilkinson Joint Regression Model:** Evaluates parental"
          " yield potential vs environmental sensitivity ($\beta_i$ slope)."
      )

      # Calculate Environmental Index (mean of each location minus grand mean)
      env_means = clean_df.groupby(env_col)[primary_trait].mean()
      grand_overall_mean = clean_df[primary_trait].mean()
      env_index = env_means - grand_overall_mean

      # Finlay-Wilkinson Regression per Line
      fw_results = []
      lines_list = clean_df[line_col].unique()

      for l in lines_list:
        sub = clean_df[clean_df[line_col] == l]
        line_env_means = sub.groupby(env_col)[primary_trait].mean()

        # Align series
        aligned = (
            pd.DataFrame({"Env_Index": env_index, "Line_Mean": line_env_means})
            .dropna()
        )

        if len(aligned) >= 2:
          slope, intercept, r_val, p_val, std_err = linregress(
              aligned["Env_Index"], aligned["Line_Mean"]
          )
          line_mean_trait = aligned["Line_Mean"].mean()

          # Classification
          if slope > 1.1:
            adapt = "Favorable Environment Specialist (High Sensitivity)"
          elif slope < 0.9:
            adapt = "Stress Tolerant / Low-Input Specialist (Robust)"
          else:
            adapt = "Broadly Adapted (Stable Across Envs)"

          fw_results.append({
              line_col: l,
              "Mean_Performance": line_mean_trait,
              "Regression_Slope_Beta": slope,
              "R2_Fit": r_val**2,
              "Adaptation_Category": adapt,
          })

      fw_df = pd.DataFrame(fw_results)

      c_fw1, c_fw2 = st.columns(2)

      with c_fw1:
        st.write("#### Finlay-Wilkinson Adaptation Scatter Chart")
        fig_fw_scatter = px.scatter(
            fw_df,
            x="Regression_Slope_Beta",
            y="Mean_Performance",
            color="Adaptation_Category",
            text=line_col,
            title="Yield Mean vs Environmental Sensitivity (Slope β)",
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
        st.write("#### Reaction Norm Plot Across Environments")
        gxe_line_df = (
            clean_df.groupby([line_col, env_col])[primary_trait]
            .mean()
            .reset_index()
        )
        fig_rn = px.line(
            gxe_line_df,
            x=env_col,
            y=primary_trait,
            color=line_col,
            markers=True,
            title="Reaction Norms Across Trial Locations",
        )
        st.plotly_chart(fig_rn, use_container_width=True)

      st.write("#### 📄 Finlay-Wilkinson Stability & Adaptation Table")
      st.dataframe(
          fw_df.sort_values(by="Mean_Performance", ascending=False),
          use_container_width=True,
      )

    # ====================================================
    # TAB 5: MULTI-TRAIT SELECTION INDEX
    # ====================================================
    with tab5:
      st.subheader("🕸️ Multi-Trait Parental Selection Index & Radar Profile")

      if len(trait_cols) > 1:
        st.write(
            "Assign relative economic weights to each target trait to calculate"
            " a unified **Parental Selection Index Score**."
        )

        weights = {}
        w_cols = st.columns(len(trait_cols))
        for idx, t in enumerate(trait_cols):
          with w_cols[idx]:
            weights[t] = st.slider(f"Weight for {t}:", -5.0, 5.0, 1.0, step=0.5)

        # Standardize traits (Z-score)
        parent_means = clean_df.groupby(line_col)[trait_cols].mean()
        z_scores = (parent_means - parent_means.mean()) / parent_means.std()

        # Compute Index Score
        index_scores = np.zeros(len(parent_means))
        for t in trait_cols:
          index_scores += z_scores[t] * weights[t]

        parent_means["Selection_Index_Score"] = index_scores
        parent_means = parent_means.sort_values(
            by="Selection_Index_Score", ascending=False
        )

        col_r1, col_r2 = st.columns(2)

        with col_r1:
          st.write("#### Top Selected Parents by Selection Index Score")
          fig_index_bar = px.bar(
              parent_means.head(10).reset_index(),
              x="Selection_Index_Score",
              y=line_col,
              orientation="h",
              color="Selection_Index_Score",
              color_continuous_scale="Viridis",
              title="Top 10 Parents Ranked by Multi-Trait Index",
              text_auto=".2f",
          )
          fig_index_bar.update_layout(
              yaxis={"categoryorder": "total ascending"}
          )
          st.plotly_chart(fig_index_bar, use_container_width=True)

        with col_r2:
          st.write("#### Multi-Trait Radar Profile (Top 5 Parents)")
          top_5_lines = parent_means.head(5).index
          radar_norm = (
              parent_means[trait_cols] - parent_means[trait_cols].min()
          ) / (
              parent_means[trait_cols].max()
              - parent_means[trait_cols].min()
              + 1e-6
          )

          fig_radar = go.Figure()
          for line in top_5_lines:
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=radar_norm.loc[line].values,
                    theta=trait_cols,
                    fill="toself",
                    name=str(line),
                )
            )
          fig_radar.update_layout(
              polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
              title="Normalized Radar Comparison",
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
    st.error(f"❌ An error occurred during data analysis: {err}")

else:
  st.info(
      "👈 Upload an Excel (.xlsx) or CSV (.csv) trial file in the sidebar to"
      " begin, or download the Sample Template to inspect the expected format."
  )
