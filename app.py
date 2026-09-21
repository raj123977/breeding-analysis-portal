import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import f
from sklearn.linear_model import Ridge
import streamlit as st

# --- Page Setup ---
st.set_page_config(
    page_title="Advanced Breeding & Line x Tester Portal",
    page_icon="🧬",
    layout="wide",
)

st.title("🌾 Quantitative Genetics & Predictive Breeding Portal")
st.markdown(
    "Perform **Multi-Location ANOVA**, **Line × Tester Analysis (GCA/SCA)**, and **Predictive Breeding Value (GEBV)** estimation."
)

# --- Sidebar File Upload ---
uploaded_file = st.sidebar.file_uploader(
    "Upload Breeding Data (.xlsx or .csv)", type=["xlsx", "csv"]
)


@st.cache_data
def load_data(file):
  if file.name.endswith(".xlsx"):
    return pd.read_excel(file)
  return pd.read_csv(file)


if uploaded_file is not None:
  df = load_data(uploaded_file)
  st.sidebar.success("File uploaded successfully!")

  cols = df.columns.tolist()

  # Identify structural columns
  st.sidebar.header("Variable Mapping")
  line_col = st.sidebar.selectbox(
      "Line Column (Female):",
      cols,
      index=cols.index("Line") if "Line" in cols else 0,
  )
  tester_col = st.sidebar.selectbox(
      "Tester Column (Male):",
      cols,
      index=cols.index("Tester") if "Tester" in cols else 0,
  )
  env_col = st.sidebar.selectbox(
      "Location/Environment Column:",
      cols,
      index=cols.index("Location_Env") if "Location_Env" in cols else 0,
  )
  rep_col = st.sidebar.selectbox(
      "Replication Column:",
      cols,
      index=cols.index("Replication") if "Replication" in cols else 0,
  )

  # Auto-generate Genotype_ID if missing
  if "Genotype_ID" not in df.columns:
    df["Genotype_ID"] = (
        df[line_col].astype(str) + " x " + df[tester_col].astype(str)
    )

  genotype_col = "Genotype_ID"
  trait_cols = st.sidebar.multiselect(
      "Target Traits:",
      [
          c
          for c in cols
          if c
          not in [
              line_col,
              tester_col,
              env_col,
              rep_col,
              genotype_col,
              "Genotype_ID",
          ]
          and not c.startswith("M_")
      ],
      default=[c for c in cols if c.startswith("Trait_")],
  )
  marker_cols = [c for c in cols if c.startswith("M_")]

  selected_trait = st.selectbox(
      "Select Primary Trait for Analysis:", trait_cols
  )

  # Tabs
  tab1, tab2, tab3, tab4, tab5 = st.tabs([
      "🏢 Multi-Location ANOVA",
      "🧬 Line × Tester (GCA/SCA)",
      "🔥 GxE Heatmaps",
      "🔮 Predictive Breeding (GEBV)",
      "📥 Download All Results",
  ])

  # ----------------------------------------------------
  # TAB 1: Multi-Location ANOVA (MET)
  # ----------------------------------------------------
  with tab1:
    st.subheader(f"Multi-Location Analysis of Variance (ANOVA): {selected_trait}")

    # Calculate MET ANOVA components
    data = df.dropna(
        subset=[env_col, genotype_col, rep_col, selected_trait]
    ).copy()

    grand_mean = data[selected_trait].mean()
    N = len(data)
    n_env = data[env_col].nunique()
    n_geno = data[genotype_col].nunique()
    n_rep = data[rep_col].nunique()

    # Sum of Squares calculations
    SS_Total = np.sum((data[selected_trait] - grand_mean) ** 2)

    env_means = data.groupby(env_col)[selected_trait].mean()
    SS_Env = n_geno * n_rep * np.sum((env_means - grand_mean) ** 2)

    rep_env_means = data.groupby([env_col, rep_col])[selected_trait].mean()
    SS_Rep_Env = (
        n_geno
        * np.sum(
            (
                rep_env_means
                - data.groupby(env_col)[selected_trait].transform("mean")
            )
            ** 2
        )
        / n_geno
    )

    geno_means = data.groupby(genotype_col)[selected_trait].mean()
    SS_Geno = n_env * n_rep * np.sum((geno_means - grand_mean) ** 2)

    cell_means = data.groupby([env_col, genotype_col])[selected_trait].mean()
    SS_GxE = n_rep * np.sum(
        (
            cell_means
            - data.groupby(env_col)[selected_trait].transform("mean")
            - data.groupby(genotype_col)[selected_trait].transform("mean")
            + grand_mean
        )
        ** 2
    )

    SS_Error = max(0, SS_Total - (SS_Env + SS_Rep_Env + SS_Geno + SS_GxE))

    # Degrees of Freedom
    df_Env = n_env - 1
    df_Rep_Env = n_env * (n_rep - 1)
    df_Geno = n_geno - 1
    df_GxE = (n_geno - 1) * (n_env - 1)
    df_Error = max(
        1,
        (N - 1) - (df_Env + df_Rep_Env + df_Geno + df_GxE),
    )

    # Mean Squares & F-tests
    MS_Env = SS_Env / df_Env if df_Env > 0 else 0
    MS_Rep_Env = SS_Rep_Env / df_Rep_Env if df_Rep_Env > 0 else 0
    MS_Geno = SS_Geno / df_Geno if df_Geno > 0 else 0
    MS_GxE = SS_GxE / df_GxE if df_GxE > 0 else 0
    MS_Error = SS_Error / df_Error if df_Error > 0 else 1e-6

    F_Geno = MS_Geno / MS_GxE if MS_GxE > 0 else MS_Geno / MS_Error
    p_Geno = f.sf(F_Geno, df_Geno, df_GxE if MS_GxE > 0 else df_Error)

    F_GxE = MS_GxE / MS_Error
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

    st.dataframe(anova_table.style.format(precision=4), use_container_width=True)

  # ----------------------------------------------------
  # TAB 2: Line x Tester Analysis (GCA & SCA)
  # ----------------------------------------------------
  with tab2:
    st.subheader("Line × Tester Combining Ability Analysis")

    lt_data = df.dropna(
        subset=[line_col, tester_col, selected_trait]
    ).copy()
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

  # ----------------------------------------------------
  # TAB 3: GxE Heatmaps & Diversity
  # ----------------------------------------------------
  with tab3:
    st.subheader("Genotype × Environment Interaction Heatmap")
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

  # ----------------------------------------------------
  # TAB 4: Predictive Breeding (GEBV)
  # ----------------------------------------------------
  with tab4:
    st.subheader("Genomic Estimated Breeding Value (GEBV) Predictions")

    if marker_cols:
      st.info(f"Using {len(marker_cols)} SNP Markers for Ridge Regression.")
      X = df[marker_cols].fillna(0)
      y = df[selected_trait].fillna(df[selected_trait].mean())

      model = Ridge(alpha=1.0)
      model.fit(X, y)
      df["Predicted_GEBV"] = model.predict(X)

      gebv_rank = (
          df.groupby(genotype_col)["Predicted_GEBV"].mean().reset_index()
      )
      gebv_rank = gebv_rank.sort_values(by="Predicted_GEBV", ascending=False)

      fig_gebv = px.bar(
          gebv_rank.head(20),
          x=genotype_col,
          y="Predicted_GEBV",
          title=f"Top 20 Ranked Genotypes by Predicted GEBV ({selected_trait})",
          color="Predicted_GEBV",
      )
      st.plotly_chart(fig_gebv, use_container_width=True)
    else:
      st.warning(
          "No marker columns starting with 'M_' detected. Phenotypic GCA + SCA"
          " Summed Value will be used as the Breeding Value proxy."
      )
      bv_df = cross_means[[line_col, tester_col, "SCA_Cross"]].merge(
          gca_lines, on=line_col
      )
      bv_df = bv_df.merge(gca_testers, on=tester_col)
      bv_df["Predicted_Breeding_Value"] = (
          overall_mean
          + bv_df["GCA_Line"]
          + bv_df["GCA_Tester"]
          + bv_df["SCA_Cross"]
      )
      st.dataframe(
          bv_df.sort_values(
              by="Predicted_Breeding_Value", ascending=False
          ),
          use_container_width=True,
      )

  # ----------------------------------------------------
  # TAB 5: Export All Results
  # ----------------------------------------------------
  with tab5:
    st.subheader("Export Results to Excel Workbook")

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
      df.to_excel(writer, sheet_name="Raw_Analyzed_Data", index=False)
      anova_table.to_excel(writer, sheet_name="MET_ANOVA", index=False)
      gca_lines.to_excel(writer, sheet_name="GCA_Lines", index=False)
      gca_testers.to_excel(writer, sheet_name="GCA_Testers", index=False)
      cross_means.to_excel(writer, sheet_name="SCA_Crosses", index=False)

    st.download_button(
        label="📥 Download Complete Quantitative Genetics Report (.xlsx)",
        data=buffer.getvalue(),
        file_name="predictive_breeding_quant_genetics_report.xlsx",
        mime="application/vnd.ms-excel",
    )

else:
  st.info("👈 Upload your Excel or CSV file in the sidebar to begin analysis.")