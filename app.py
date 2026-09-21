import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import f
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
import streamlit as st

# --- Page Setup & Modern Styling ---
st.set_page_config(
    page_title="Enterprise Predictive Breeding Portal",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for UI polish
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #2e7d32;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f1f3f5;
        border-radius: 6px 6px 0px 0px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2e7d32 !important;
        color: white !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.title("🌾 Quantitative Genetics & Predictive Breeding Portal")
st.markdown(
    "Advanced Analytics Engine for **Multi-Environment Trials (MET)**, **Line ×"
    " Tester Combining Ability**, **GxE Stability**, and **Genomic Selection"
    " (GEBV)**."
)

# --- File Upload Section ---
uploaded_file = st.sidebar.file_uploader(
    "Upload Trial Dataset (.xlsx or .csv)", type=["xlsx", "csv"]
)


@st.cache_data
def load_data(file):
  if file.name.endswith(".xlsx"):
    return pd.read_excel(file)
  return pd.read_csv(file)


if uploaded_file is not None:
  try:
    df = load_data(uploaded_file)
    st.sidebar.success("Data successfully loaded!")

    cols = df.columns.tolist()

    # Smart Column Matcher
    def get_default_index(target_names, col_list, fallback=0):
      for name in target_names:
        for idx, col in enumerate(col_list):
          if name.lower() in col.lower():
            return idx
      return fallback

    line_idx = get_default_index(["line", "female", "parent1"], cols, 0)
    tester_idx = get_default_index(
        ["tester", "male", "parent2"], cols, min(1, len(cols) - 1)
    )
    env_idx = get_default_index(
        ["location", "loc", "env", "trial"], cols, min(2, len(cols) - 1)
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

    # Generate Genotype_ID
    if "Genotype_ID" not in df.columns:
      df["Genotype_ID"] = (
          df[line_col].astype(str) + " × " + df[tester_col].astype(str)
      )

    genotype_col = "Genotype_ID"

    available_traits = [
        c
        for c in cols
        if c not in [line_col, tester_col, env_col, rep_col, genotype_col]
        and not c.startswith("M_")
    ]
    default_traits = [c for c in available_traits if "trait" in c.lower()] or (
        [available_traits[0]] if available_traits else []
    )

    trait_cols = st.sidebar.multiselect(
        "Select Target Traits:", available_traits, default=default_traits
    )
    marker_cols = [c for c in cols if c.startswith("M_")]

    if not trait_cols:
      st.warning("⚠️ Please select at least one numerical trait column.")
      st.stop()

    selected_trait = st.selectbox("🎯 Primary Focus Trait:", trait_cols)
    df[selected_trait] = pd.to_numeric(df[selected_trait], errors="coerce")

    # Clean working dataset
    clean_df = df.dropna(
        subset=[env_col, genotype_col, rep_col, selected_trait]
    ).copy()

    # --- Top Metric Dashboard Cards ---
    st.markdown("### 📊 Dataset High-Level Summary")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Observations", len(clean_df))
    m2.metric("Genotypes (Hybrids)", clean_df[genotype_col].nunique())
    m3.metric("Environments", clean_df[env_col].nunique())
    m4.metric("Replications", clean_df[rep_col].nunique())

    # Approximate Heritability (Broad-Sense H^2)
    var_g = clean_df.groupby(genotype_col)[selected_trait].mean().var()
    var_total = clean_df[selected_trait].var()
    approx_h2 = (
        min(max(var_g / var_total, 0.05), 0.95) if var_total > 0 else 0.0
    )
    m5.metric("Approx. Heritability (H²)", f"{approx_h2:.2f}")

    # --- Tab Navigation ---
    tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🌐 3D Trial Space",
        "🏢 MET ANOVA & Performance",
        "🧬 Line × Tester (GCA/SCA)",
        "📈 GxE Reaction Norms",
        "🔮 Predictive Selection (GEBV)",
        "🕸️ Multi-Trait Profile",
        "📥 Export Reports",
    ])

    # ====================================================
    # TAB 0: 3D Trial Space Overview
    # ====================================================
    with tab0:
      st.subheader(
          f"3D Multi-Environment Trial Visualization: {selected_trait}"
      )
      st.write(
          "Explore the complete trial landscape across Genotypes, Locations,"
          " and Observed Trait Values."
      )

      # 3D Scatter Visual
      fig_3d_trial = px.scatter_3d(
          clean_df,
          x=env_col,
          y=genotype_col,
          z=selected_trait,
          color=selected_trait,
          size=selected_trait,
          color_continuous_scale="Turbo",
          title=f"3D Scatter: Environment vs Genotype vs {selected_trait}",
          opacity=0.85,
      )
      fig_3d_trial.update_layout(
          height=650, scene=dict(zaxis_title=selected_trait)
      )
      st.plotly_chart(fig_3d_trial, use_container_width=True)

      with st.expander("📄 View Full Raw Trial Data Table"):
        st.dataframe(clean_df, use_container_width=True)

    # ====================================================
    # TAB 1: Multi-Location ANOVA & Distribution
    # ====================================================
    with tab1:
      st.subheader("Multi-Environment Trial (MET) ANOVA & Distribution")

      v_col1, v_col2 = st.columns(2)

      with v_col1:
        st.write("#### Phenotypic Distribution Across Locations")
        fig_violin = px.violin(
            clean_df,
            x=env_col,
            y=selected_trait,
            color=env_col,
            box=True,
            points="all",
            color_discrete_sequence=px.colors.qualitative.Dark2,
            title=f"Violin & Box Plot of {selected_trait} per Location",
        )
        st.plotly_chart(fig_violin, use_container_width=True)

      with v_col2:
        st.write("#### Location Means & Standard Errors")
        env_summary = (
            clean_df.groupby(env_col)[selected_trait]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        env_summary["se"] = env_summary["std"] / np.sqrt(
            env_summary["count"]
        )

        fig_env_bar = px.bar(
            env_summary,
            x=env_col,
            y="mean",
            error_y="se",
            color="mean",
            color_continuous_scale="Viridis",
            title=f"Mean {selected_trait} by Location (with Standard Error)",
            text_auto=".2f",
        )
        st.plotly_chart(fig_env_bar, use_container_width=True)

      # ANOVA Calculations
      st.write("### Multi-Location ANOVA Table")
      grand_mean = clean_df[selected_trait].mean()
      N = len(clean_df)
      n_env = clean_df[env_col].nunique()
      n_geno = clean_df[genotype_col].nunique()
      n_rep = clean_df[rep_col].nunique()

      clean_df["E_mean"] = clean_df.groupby(env_col)[
          selected_trait
      ].transform("mean")
      clean_df["G_mean"] = clean_df.groupby(genotype_col)[
          selected_trait
      ].transform("mean")
      clean_df["R_E_mean"] = clean_df.groupby([env_col, rep_col])[
          selected_trait
      ].transform("mean")
      clean_df["GE_mean"] = clean_df.groupby([env_col, genotype_col])[
          selected_trait
      ].transform("mean")

      SS_Total = np.sum((clean_df[selected_trait] - grand_mean) ** 2)
      SS_Env = np.sum((clean_df["E_mean"] - grand_mean) ** 2)
      SS_Rep_Env = np.sum((clean_df["R_E_mean"] - clean_df["E_mean"]) ** 2)
      SS_Geno = np.sum((clean_df["G_mean"] - grand_mean) ** 2)
      SS_GxE = np.sum(
          (
              clean_df["GE_mean"]
              - clean_df["E_mean"]
              - clean_df["G_mean"]
              + grand_mean
          )
          ** 2
      )
      SS_Error = max(0.0, SS_Total - (SS_Env + SS_Rep_Env + SS_Geno + SS_GxE))

      df_Env = max(1, n_env - 1)
      df_Rep_Env = max(1, n_env * (n_rep - 1))
      df_Geno = max(1, n_geno - 1)
      df_GxE = max(1, (n_geno - 1) * (n_env - 1))
      df_Error = max(1, (N - 1) - (df_Env + df_Rep_Env + df_Geno + df_GxE))

      MS_Env, MS_Rep_Env = SS_Env / df_Env, SS_Rep_Env / df_Rep_Env
      MS_Geno, MS_GxE = SS_Geno / df_Geno, SS_GxE / df_GxE
      MS_Error = SS_Error / df_Error if df_Error > 0 else 1e-6

      F_Geno = MS_Geno / MS_GxE if MS_GxE > 0 else MS_Geno / MS_Error
      p_Geno = f.sf(F_Geno, df_Geno, df_GxE if MS_GxE > 0 else df_Error)
      F_GxE = MS_GxE / MS_Error if MS_Error > 0 else 0
      p_GxE = f.sf(F_GxE, df_GxE, df_Error)

      def get_stars(p):
        if np.isnan(p):
          return ""
        if p < 0.001:
          return "***"
        if p < 0.01:
          return "**"
        if p < 0.05:
          return "*"
        return "ns"

      anova_df = pd.DataFrame({
          "Source of Variation": [
              "Location/Environment (E)",
              "Replication within Env R(E)",
              "Genotype (G)",
              "Genotype × Environment (GxE)",
              "Residual Error",
              "Total",
          ],
          "DF": [df_Env, df_Rep_Env, df_Geno, df_GxE, df_Error, N - 1],
          "Sum of Squares (SS)": [
              SS_Env,
              SS_Rep_Env,
              SS_Geno,
              SS_GxE,
              SS_Error,
              SS_Total,
          ],
          "Mean Square (MS)": [
              MS_Env,
              MS_Rep_Env,
              MS_Geno,
              MS_GxE,
              MS_Error,
              np.nan,
          ],
          "F-Value": [np.nan, np.nan, F_Geno, F_GxE, np.nan, np.nan],
          "p-Value": [np.nan, np.nan, p_Geno, p_GxE, np.nan, np.nan],
          "Significance": [
              "",
              "",
              get_stars(p_Geno),
              get_stars(p_GxE),
              "",
              "",
          ],
      })
      st.dataframe(
          anova_df.style.format(
              {"Sum of Squares (SS)": "{:.2f}", "Mean Square (MS)": "{:.2f}", "F-Value": "{:.2f}", "p-Value": "{:.4f}"},
              na_rep="",
          ),
          use_container_width=True,
      )

    # ====================================================
    # TAB 2: Line x Tester (GCA, SCA & 3D Surface)
    # ====================================================
    with tab2:
      st.subheader("Line × Tester Combining Ability & 3D Heterosis Analysis")

      lt_df = clean_df.dropna(subset=[line_col, tester_col, selected_trait])
      overall_mean = lt_df[selected_trait].mean()

      # GCA Calculation
      gca_lines = (
          lt_df.groupby(line_col)[selected_trait].mean() - overall_mean
      ).reset_index()
      gca_lines.columns = [line_col, "GCA_Line"]

      gca_testers = (
          lt_df.groupby(tester_col)[selected_trait].mean() - overall_mean
      ).reset_index()
      gca_testers.columns = [tester_col, "GCA_Tester"]

      # Crosses & SCA
      cross_means = (
          lt_df.groupby([line_col, tester_col])[selected_trait]
          .mean()
          .reset_index()
      )
      cross_means = cross_means.merge(gca_lines, on=line_col).merge(
          gca_testers, on=tester_col
      )
      cross_means["SCA_Cross"] = (
          cross_means[selected_trait]
          - overall_mean
          - cross_means["GCA_Line"]
          - cross_means["GCA_Tester"]
      )
      cross_means["Cross_Name"] = (
          cross_means[line_col].astype(str)
          + " × "
          + cross_means[tester_col].astype(str)
      )

      # --- 3D Surface Plot of Line x Tester Interaction ---
      st.write("#### 3D Surface Plot: Line × Tester Trait Surface")
      sca_pivot = cross_means.pivot(
          index=line_col, columns=tester_col, values=selected_trait
      )

      fig_3d_lt = go.Figure(
          data=[
              go.Surface(
                  z=sca_pivot.values,
                  x=sca_pivot.columns.astype(str),
                  y=sca_pivot.index.astype(str),
                  colorscale="Viridis",
              )
          ]
      )
      fig_3d_lt.update_layout(
          title="3D Hybrid Performance Surface (Female Line x Male Tester)",
          scene=dict(
              xaxis_title="Tester (Male)",
              yaxis_title="Line (Female)",
              zaxis_title=selected_trait,
          ),
          height=600,
      )
      st.plotly_chart(fig_3d_lt, use_container_width=True)

      # Diverging GCA Charts
      gca_col1, gca_col2 = st.columns(2)
      with gca_col1:
        gca_lines["Type"] = np.where(
            gca_lines["GCA_Line"] >= 0, "Positive", "Negative"
        )
        fig_gcal = px.bar(
            gca_lines.sort_values(by="GCA_Line"),
            x="GCA_Line",
            y=line_col,
            orientation="h",
            color="Type",
            color_discrete_map={"Positive": "#2ca02c", "Negative": "#d62728"},
            title="Line GCA Effects (Female Additive)",
            text_auto=".2f",
        )
        fig_gcal.add_vline(x=0, line_dash="dash")
        st.plotly_chart(fig_gcal, use_container_width=True)

      with gca_col2:
        gca_testers["Type"] = np.where(
            gca_testers["GCA_Tester"] >= 0, "Positive", "Negative"
        )
        fig_gcat = px.bar(
            gca_testers.sort_values(by="GCA_Tester"),
            x="GCA_Tester",
            y=tester_col,
            orientation="h",
            color="Type",
            color_discrete_map={"Positive": "#1f77b4", "Negative": "#ff7f0e"},
            title="Tester GCA Effects (Male Additive)",
            text_auto=".2f",
        )
        fig_gcat.add_vline(x=0, line_dash="dash")
        st.plotly_chart(fig_gcat, use_container_width=True)

      # SCA Heatmap
      st.write("#### Specific Combining Ability (SCA) Matrix Heatmap")
      sca_matrix = cross_means.pivot(
          index=line_col, columns=tester_col, values="SCA_Cross"
      )
      fig_sca_hm = px.imshow(
          sca_matrix,
          text_auto=".2f",
          color_continuous_scale="RdBu_r",
          title="SCA Matrix (Specific Hybrid Combinations)",
      )
      st.plotly_chart(fig_sca_hm, use_container_width=True)

      with st.expander("📄 View Full Line × Tester Statistics Table"):
        st.dataframe(
            cross_means[[
                line_col,
                tester_col,
                selected_trait,
                "GCA_Line",
                "GCA_Tester",
                "SCA_Cross",
            ]],
            use_container_width=True,
        )

    # ====================================================
    # TAB 3: GxE Reaction Norms & 3D Stability Space
    # ====================================================
    with tab3:
      st.subheader("Genotype × Environment Interaction & Stability Analysis")

      gxe_df = (
          clean_df.groupby([genotype_col, env_col])[selected_trait]
          .mean()
          .reset_index()
      )

      st.write("#### Reaction Norm Plot (Finlay-Wilkinson Adaptation)")
      fig_rn = px.line(
          gxe_df,
          x=env_col,
          y=selected_trait,
          color=genotype_col,
          markers=True,
          title=f"Stability Profiles Across Locations: {selected_trait}",
      )
      st.plotly_chart(fig_rn, use_container_width=True)

      # 3D Stability Space
      st.write("#### 3D Genotype Stability & Performance Space")
      stability_df = (
          gxe_df.groupby(genotype_col)[selected_trait]
          .agg(["mean", "std", "var"])
          .reset_index()
      )
      stability_df["CV_pct"] = (
          stability_df["std"] / stability_df["mean"]
      ) * 100

      fig_3d_stab = px.scatter_3d(
          stability_df,
          x="mean",
          y="std",
          z="CV_pct",
          color="mean",
          size="mean",
          hover_name=genotype_col,
          color_continuous_scale="Plasma",
          title=(
              "3D Stability Space: Mean Performance (X) vs Standard"
              " Deviation (Y) vs CV% (Z)"
          ),
      )
      fig_3d_stab.update_layout(height=600)
      st.plotly_chart(fig_3d_stab, use_container_width=True)

      with st.expander("📄 View Genotype Stability Summary Table"):
        st.dataframe(
            stability_df.sort_values(by="mean", ascending=False),
            use_container_width=True,
        )

    # ====================================================
    # TAB 4: Predictive Breeding (GEBV & 3D Genomic PCA)
    # ====================================================
    with tab4:
      st.subheader(
          "Genomic Estimated Breeding Value (GEBV) & Genomic Space"
      )

      if marker_cols:
        st.info(
            f"🧬 Found {len(marker_cols)} SNP Markers. Running Ridge"
            " Regression & 3D PCA Population Structure Analysis..."
        )

        X = clean_df[marker_cols].fillna(0)
        y = clean_df[selected_trait].fillna(clean_df[selected_trait].mean())

        # Fit Ridge GEBV
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        clean_df["Predicted_GEBV"] = model.predict(X)

        # 3D PCA on Markers
        pca = PCA(n_components=3)
        pca_coords = pca.fit_transform(X)
        clean_df["PC1"] = pca_coords[:, 0]
        clean_df["PC2"] = pca_coords[:, 1]
        clean_df["PC3"] = pca_coords[:, 2]

        c1, c2 = st.columns(2)
        with c1:
          st.write("#### Observed vs Predicted GEBV Model Accuracy")
          fig_acc = px.scatter(
              clean_df,
              x=selected_trait,
              y="Predicted_GEBV",
              color=genotype_col,
              trendline="ols",
              title="GEBV Prediction Accuracy Scatter",
          )
          st.plotly_chart(fig_acc, use_container_width=True)

        with c2:
          st.write("#### Top Selected Candidates by GEBV")
          gebv_rank = (
              clean_df.groupby(genotype_col)["Predicted_GEBV"]
              .mean()
              .reset_index()
              .sort_values(by="Predicted_GEBV", ascending=False)
              .head(15)
          )

          fig_gebv_bar = px.bar(
              gebv_rank,
              x="Predicted_GEBV",
              y=genotype_col,
              orientation="h",
              color="Predicted_GEBV",
              color_continuous_scale="Viridis",
              title="Top 15 Breeding Candidates (GEBV)",
              text_auto=".2f",
          )
          fig_gebv_bar.update_layout(
              yaxis={"categoryorder": "total ascending"}
          )
          st.plotly_chart(fig_gebv_bar, use_container_width=True)

        # 3D Genomic PCA Space
        st.write("#### 3D Genomic PCA Space Colored by Predicted GEBV")
        fig_3d_pca = px.scatter_3d(
            clean_df,
            x="PC1",
            y="PC2",
            z="PC3",
            color="Predicted_GEBV",
            hover_name=genotype_col,
            color_continuous_scale="Spectral",
            title=(
                "3D Population Structure (PC1 x PC2 x PC3) Overlayed with GEBV"
            ),
        )
        fig_3d_pca.update_layout(height=600)
        st.plotly_chart(fig_3d_pca, use_container_width=True)

      else:
        st.warning(
            "⚠️ No SNP columns (starting with 'M_') were found in your"
            " dataset. Showing phenotypic rank candidates instead."
        )
        pheno_rank = (
            clean_df.groupby(genotype_col)[selected_trait]
            .mean()
            .reset_index()
            .sort_values(by=selected_trait, ascending=False)
            .head(15)
        )

        fig_pheno = px.bar(
            pheno_rank,
            x=selected_trait,
            y=genotype_col,
            orientation="h",
            color=selected_trait,
            color_continuous_scale="Cividis",
            title=f"Top 15 Genotypes by Observed Mean {selected_trait}",
            text_auto=".2f",
        )
        fig_pheno.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_pheno, use_container_width=True)

    # ====================================================
    # TAB 5: Multi-Trait Profile & Radar
    # ====================================================
    with tab5:
      st.subheader("Multi-Trait Correlations & Radar Profile")

      if len(trait_cols) > 1:
        tc1, tc2 = st.columns(2)

        with tc1:
          st.write("#### Trait Correlation Heatmap")
          corr_matrix = clean_df[trait_cols].corr()
          fig_corr = px.imshow(
              corr_matrix,
              text_auto=".2f",
              color_continuous_scale="Coolwarm",
              title="Pairwise Trait Correlation Matrix",
          )
          st.plotly_chart(fig_corr, use_container_width=True)

        with tc2:
          st.write("#### Top Genotypes Radar Comparison")
          top_5_genos = (
              clean_df.groupby(genotype_col)[selected_trait]
              .mean()
              .nlargest(5)
              .index
          )
          radar_df = (
              clean_df[clean_df[genotype_col].isin(top_5_genos)]
              .groupby(genotype_col)[trait_cols]
              .mean()
          )

          # Normalize 0-1 for radar overlay
          radar_norm = (radar_df - radar_df.min()) / (
              radar_df.max() - radar_df.min() + 1e-6
          )

          fig_radar = go.Figure()
          for g in radar_norm.index:
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
              title="Normalized Multi-Trait Profile (Top 5 Genotypes)",
          )
          st.plotly_chart(fig_radar, use_container_width=True)
      else:
        st.info("Select multiple target traits in the sidebar to enable Multi-Trait Profiling.")

    # ====================================================
    # TAB 6: Export & Excel Report
    # ====================================================
    with tab6:
      st.subheader("📥 Export Complete Quantitative Analysis Report")
      buffer = io.BytesIO()
      with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        clean_df.to_excel(writer, sheet_name="Clean_Trial_Data", index=False)
        if "anova_df" in locals():
          anova_df.to_excel(writer, sheet_name="MET_ANOVA", index=False)
        if "cross_means" in locals():
          cross_means.to_excel(writer, sheet_name="Line_x_Tester", index=False)
        if "stability_df" in locals():
          stability_df.to_excel(
              writer, sheet_name="Genotype_Stability", index=False
          )

      st.download_button(
          label="📥 Download Complete Quantitative Genetics Excel Report (.xlsx)",
          data=buffer.getvalue(),
          file_name="predictive_breeding_complete_report.xlsx",
          mime="application/vnd.ms-excel",
      )

  except Exception as main_err:
    st.error(f"❌ An unexpected error occurred: {main_err}")

else:
  st.info(
      "👈 Please upload a trial dataset (.xlsx or .csv) in the left sidebar to"
      " launch the analytics portal."
  )
