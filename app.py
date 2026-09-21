import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import f
from sklearn.linear_model import Ridge
import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="Predictive Breeding & Quantitative Genetics Portal",
    page_icon="🌾",
    layout="wide",
)

st.title("🌾 Quantitative Genetics & Predictive Breeding Portal")
st.markdown(
    "Upload your breeding trial data to run **Multi-Location ANOVA**, **Line"
    " × Tester Analysis (GCA/SCA)**, and **Predictive Breeding Value (GEBV)**"
    " modeling."
)

# --- File Upload Section ---
uploaded_file = st.sidebar.file_uploader(
    "Upload Breeding Data (.xlsx or .csv)", type=["xlsx", "csv"]
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

    st.subheader("1. Variable Mapping & Verification")
    st.write("Data Preview:", df.head(5))

    cols = df.columns.tolist()

    # Sidebar Variable Selection
    st.sidebar.header("Variable Mapping")

    # Safe default column detection
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

    line_col = st.sidebar.selectbox("Line Column (Female):", cols, index=line_idx)
    tester_col = st.sidebar.selectbox(
        "Tester Column (Male):", cols, index=tester_idx
    )
    env_col = st.sidebar.selectbox(
        "Location / Environment Column:", cols, index=env_idx
    )
    rep_col = st.sidebar.selectbox("Replication Column:", cols, index=rep_idx)

    # Auto-generate Genotype_ID if not present
    if "Genotype_ID" not in df.columns:
      df["Genotype_ID"] = (
          df[line_col].astype(str) + " x " + df[tester_col].astype(str)
      )

    genotype_col = "Genotype_ID"

    # Identify Trait and Marker Columns
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
        "Target Traits:", available_traits, default=default_traits
    )
    marker_cols = [c for c in cols if c.startswith("M_")]

    if not trait_cols:
      st.warning(
          "Please select at least one numerical trait column from the sidebar."
      )
      st.stop()

    selected_trait = st.selectbox("Primary Trait for Deep Analysis:", trait_cols)

    # Analysis Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏢 Multi-Location ANOVA",
        "🧬 Line × Tester (GCA/SCA)",
        "🔥 GxE Heatmaps",
        "🔮 Predictive Breeding (GEBV)",
        "📥 Export Results",
    ])

    # ----------------------------------------------------
    # TAB 1: Multi-Location ANOVA (Fixed Vector Math)
    # ----------------------------------------------------
    with tab1:
      st.subheader(
          f"Multi-Location Analysis of Variance (MET ANOVA): {selected_trait}"
      )
      try:
        data = df.dropna(
            subset=[env_col, genotype_col, rep_col, selected_trait]
        ).copy()

        # Clean numerical conversion
        data[selected_trait] = pd.to_numeric(
            data[selected_trait], errors="coerce"
        )
        data = data.dropna(subset=[selected_trait])

        grand_mean = data[selected_trait].mean()
        N = len(data)
        n_env = data[env_col].nunique()
        n_geno = data[genotype_col].nunique()
        n_rep = data[rep_col].nunique()

        # Vectorized Row-Level Calculations (Index-Safe)
        data["E_mean"] = data.groupby(env_col)[selected_trait].transform(
            "mean"
        )
        data["G_mean"] = data.groupby(genotype_col)[selected_trait].transform(
            "mean"
        )
        data["R_E_mean"] = data.groupby([env_col, rep_col])[
            selected_trait
        ].transform("mean")
        data["GE_mean"] = data.groupby([env_col, genotype_col])[
            selected_trait
        ].transform("mean")

        SS_Total = np.sum((data[selected_trait] - grand_mean) ** 2)
        SS_Env = np.sum((data["E_mean"] - grand_mean) ** 2)
        SS_Rep_Env = np.sum((data["R_E_mean"] - data["E_mean"]) ** 2)
        SS_Geno = np.sum((data["G_mean"] - grand_mean) ** 2)
        SS_GxE = np.sum(
            (data["GE_mean"] - data["E_mean"] - data["G_mean"] + grand_mean)
            ** 2
        )
        SS_Error = max(
            0.0, SS_Total - (SS_Env + SS_Rep_Env + SS_Geno + SS_GxE)
        )

        df_Env = max(1, n_env - 1)
        df_Rep_Env = max(1, n_env * (n_rep - 1))
        df_Geno = max(1, n_geno - 1)
        df_GxE = max(1, (n_geno - 1) * (n_env - 1))
        df_Error = max(
            1, (N - 1) - (df_Env + df_Rep_Env + df_Geno + df_GxE)
        )

        MS_Env = SS_Env / df_Env
        MS_Rep_Env = SS_Rep_Env / df_Rep_Env
        MS_Geno = SS_Geno / df_Geno
        MS_GxE = SS_GxE / df_GxE
        MS_Error = SS_Error / df_Error if df_Error > 0 else 1e-6

        F_Geno = MS_Geno / MS_GxE if MS_GxE > 0 else MS_Geno / MS_Error
        p_Geno = f.sf(F_Geno, df_Geno, df_GxE if MS_GxE > 0 else df_Error)

        F_GxE = MS_GxE / MS_Error if MS_Error > 0 else 0
        p_GxE = f.sf(F_GxE, df_GxE, df_Error)

        anova_table = pd.DataFrame({
            "Source of Variation": [
                "Location/Environment (E)",
                "Replication within Env R(E)",
                "Genotype (G)",
                "Genotype × Environment (GxE)",
                "Residual Error",
                "Total",
            ],
            "DF": [
                df_Env,
                df_Rep_Env,
                df_Geno,
                df_GxE,
                df_Error,
                N - 1,
            ],
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
            "p-value": [np.nan, np.nan, p_Geno, p_GxE, np.nan, np.nan],
        })

        st.dataframe(
            anova_table.style.format(precision=4, na_rep=""),
            use_container_width=True,
        )

      except Exception as e:
        st.error(
            f"Could not calculate ANOVA: {e}. Check if Replication and"
            " Location columns contain valid numerical/categorical groups."
        )

    # ----------------------------------------------------
    # TAB 2: Line x Tester (GCA & SCA)
    # ----------------------------------------------------
    with tab2:
      st.subheader("Line × Tester Combining Ability Analysis")
      try:
        lt_data = df.dropna(
            subset=[line_col, tester_col, selected_trait]
        ).copy()
        lt_data[selected_trait] = pd.to_numeric(
            lt_data[selected_trait], errors="coerce"
        )
        lt_data = lt_data.dropna(subset=[selected_trait])

        overall_mean = lt_data[selected_trait].mean()

        # GCA Lines
        line_means = lt_data.groupby(line_col)[selected_trait].mean()
        gca_lines = (line_means - overall_mean).reset_index()
        gca_lines.columns = [line_col, "GCA_Line"]

        # GCA Testers
        tester_means = lt_data.groupby(tester_col)[selected_trait].mean()
        gca_testers = (tester_means - overall_mean).reset_index()
        gca_testers.columns = [tester_col, "GCA_Tester"]

        # SCA Crosses
        cross_means = (
            lt_data.groupby([line_col, tester_col])[selected_trait]
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

        col1, col2 = st.columns(2)
        with col1:
          st.write("### GCA Effects: Lines (Female Parents)")
          st.dataframe(
              gca_lines.sort_values(by="GCA_Line", ascending=False),
              use_container_width=True,
          )
          fig_gca_l = px.bar(
              gca_lines,
              x=line_col,
              y="GCA_Line",
              title="Line GCA Effects",
              color="GCA_Line",
          )
          st.plotly_chart(fig_gca_l, use_container_width=True)

        with col2:
          st.write("### GCA Effects: Testers (Male Parents)")
          st.dataframe(
              gca_testers.sort_values(by="GCA_Tester", ascending=False),
              use_container_width=True,
          )
          fig_gca_t = px.bar(
              gca_testers,
              x=tester_col,
              y="GCA_Tester",
              title="Tester GCA Effects",
              color="GCA_Tester",
          )
          st.plotly_chart(fig_gca_t, use_container_width=True)

        st.write("### SCA Effects: Crosses (Line × Tester Hybrids)")
        sca_pivot = cross_means.pivot(
            index=line_col, columns=tester_col, values="SCA_Cross"
        )
        fig_sca = px.imshow(
            sca_pivot,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            title="SCA Heatmap (Line x Tester Interactions)",
        )
        st.plotly_chart(fig_sca, use_container_width=True)

      except Exception as e:
        st.error(
            f"Could not calculate Line × Tester effects: {e}. Please ensure Line"
            " and Tester columns are properly mapped in the sidebar."
        )

    # ----------------------------------------------------
    # TAB 3: GxE Heatmaps
    # ----------------------------------------------------
    with tab3:
      st.subheader("Genotype × Environment Interaction Heatmap")
      try:
        gxe_pivot = df.pivot_table(
            index=genotype_col,
            columns=env_col,
            values=selected_trait,
            aggfunc="mean",
        )
        fig_gxe = px.imshow(
            gxe_pivot,
            color_continuous_scale="Viridis",
            aspect="auto",
            title=f"Mean Performance of {selected_trait} across Environments",
        )
        st.plotly_chart(fig_gxe, use_container_width=True)
      except Exception as e:
        st.error(f"Unable to generate GxE Heatmap: {e}")

    # ----------------------------------------------------
    # TAB 4: Predictive Breeding (GEBV)
    # ----------------------------------------------------
    with tab4:
      st.subheader("Genomic Estimated Breeding Value (GEBV) Predictions")

      if marker_cols:
        try:
          st.info(
              f"Detected {len(marker_cols)} SNP markers starting with 'M_'."
              " Running Ridge Regression (GBLUP Proxy)..."
          )
          X = df[marker_cols].fillna(0)
          y = pd.to_numeric(df[selected_trait], errors="coerce").fillna(
              df[selected_trait].mean()
          )

          model = Ridge(alpha=1.0)
          model.fit(X, y)
          df["Predicted_GEBV"] = model.predict(X)

          gebv_rank = (
              df.groupby(genotype_col)["Predicted_GEBV"].mean().reset_index()
          )
          gebv_rank = gebv_rank.sort_values(
              by="Predicted_GEBV", ascending=False
          )

          fig_gebv = px.bar(
              gebv_rank.head(20),
              x=genotype_col,
              y="Predicted_GEBV",
              title=(
                  f"Top 20 Ranked Genotypes by Predicted GEBV ({selected_trait})"
              ),
              color="Predicted_GEBV",
          )
          st.plotly_chart(fig_gebv, use_container_width=True)
        except Exception as e:
          st.error(f"Error fitting GEBV model: {e}")
      else:
        st.info(
            "No SNP marker columns starting with 'M_' were found. Showing"
            " Genotype Phenotypic Means ranking instead."
        )
        mean_rank = (
            df.groupby(genotype_col)[selected_trait]
            .mean()
            .reset_index()
            .sort_values(by=selected_trait, ascending=False)
        )
        fig_rank = px.bar(
            mean_rank.head(20),
            x=genotype_col,
            y=selected_trait,
            title=f"Top 20 Ranked Genotypes by Observed Mean ({selected_trait})",
        )
        st.plotly_chart(fig_rank, use_container_width=True)

    # ----------------------------------------------------
    # TAB 5: Export All Results
    # ----------------------------------------------------
    with tab5:
      st.subheader("Export Results to Excel Workbook")
      buffer = io.BytesIO()
      with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Analyzed_Data", index=False)
        if "anova_table" in locals():
          anova_table.to_excel(writer, sheet_name="MET_ANOVA", index=False)
        if "gca_lines" in locals():
          gca_lines.to_excel(writer, sheet_name="GCA_Lines", index=False)
          gca_testers.to_excel(writer, sheet_name="GCA_Testers", index=False)
          cross_means.to_excel(writer, sheet_name="SCA_Crosses", index=False)

      st.download_button(
          label="📥 Download Complete Breeding Analysis Report (.xlsx)",
          data=buffer.getvalue(),
          file_name="predictive_breeding_analysis_report.xlsx",
          mime="application/vnd.ms-excel",
      )

  except Exception as main_err:
    st.error(
        f"An error occurred while reading the dataset: {main_err}. Please"
        " verify that your uploaded Excel/CSV file is properly formatted."
    )

else:
  st.info(
      "👈 Upload an Excel (.xlsx) or CSV (.csv) file in the left sidebar to"
      " begin analysis."
  )
