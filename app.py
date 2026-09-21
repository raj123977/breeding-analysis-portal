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
    page_title="Predictive Breeding & Quantitative Genetics Portal",
    page_icon="🌾",
    layout="wide",
)

st.title("🌾 Quantitative Genetics & Predictive Breeding Portal")
st.markdown(
    "Upload your breeding trial dataset to generate **Location Boxplots**, **GCA/SCA Diverging Charts**, **Reaction Norm Plots**, and **GEBV Predictions**."
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

    st.subheader("1. Variable Setup")
    st.write("Data Preview:", df.head(4))

    cols = df.columns.tolist()

    # Smart Column Index Matcher
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

    st.sidebar.header("Variable Mapping")
    line_col = st.sidebar.selectbox("Line Column (Female):", cols, index=line_idx)
    tester_col = st.sidebar.selectbox(
        "Tester Column (Male):", cols, index=tester_idx
    )
    env_col = st.sidebar.selectbox(
        "Location / Environment Column:", cols, index=env_idx
    )
    rep_col = st.sidebar.selectbox("Replication Column:", cols, index=rep_idx)

    # Auto-generate Genotype_ID if missing
    if "Genotype_ID" not in df.columns:
      df["Genotype_ID"] = (
          df[line_col].astype(str) + " x " + df[tester_col].astype(str)
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
        "Target Traits:", available_traits, default=default_traits
    )
    marker_cols = [c for c in cols if c.startswith("M_")]

    if not trait_cols:
      st.warning("Select at least one numeric trait column in the sidebar.")
      st.stop()

    selected_trait = st.selectbox("Primary Trait for Visualization:", trait_cols)

    # Convert selected trait to float safely
    df[selected_trait] = pd.to_numeric(df[selected_trait], errors="coerce")

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏢 Multi-Location ANOVA",
        "🧬 Line × Tester (GCA / SCA)",
        "📈 GxE Reaction Norms",
        "🔮 Predictive Breeding (GEBV)",
        "📥 Export Results",
    ])

    # ----------------------------------------------------
    # TAB 1: Multi-Location ANOVA & Visuals
    # ----------------------------------------------------
    with tab1:
      st.subheader(f"Multi-Location Performance & ANOVA: {selected_trait}")
      try:
        data = df.dropna(
            subset=[env_col, genotype_col, rep_col, selected_trait]
        ).copy()

        # Visual 1: Environment Trait Distribution Boxplot
        fig_box = px.box(
            data,
            x=env_col,
            y=selected_trait,
            color=env_col,
            points="all",
            title=f"Distribution of {selected_trait} Across Trial Environments",
            labels={
                selected_trait: f"{selected_trait} Value",
                env_col: "Environment / Location",
            },
        )
        st.plotly_chart(fig_box, use_container_width=True)

        # Visual 2: Mean Performance by Environment
        env_means = (
            data.groupby(env_col)[selected_trait]
            .mean()
            .reset_index()
            .sort_values(by=selected_trait, ascending=False)
        )
        fig_env_bar = px.bar(
            env_means,
            x=env_col,
            y=selected_trait,
            color=selected_trait,
            color_continuous_scale="Viridis",
            title=f"Mean {selected_trait} by Location",
            text_auto=".2f",
        )
        st.plotly_chart(fig_env_bar, use_container_width=True)

        # ANOVA Table Computation
        grand_mean = data[selected_trait].mean()
        N = len(data)
        n_env = data[env_col].nunique()
        n_geno = data[genotype_col].nunique()
        n_rep = data[rep_col].nunique()

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

        st.write("### Multi-Location ANOVA Table")
        anova_table = pd.DataFrame({
            "Source of Variation": [
                "Location / Environment (E)",
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
        st.error(f"Error calculating Multi-Location statistics: {e}")

    # ----------------------------------------------------
    # TAB 2: Line x Tester (Diverging GCA & SCA Charts)
    # ----------------------------------------------------
    with tab2:
      st.subheader("Combining Ability Analysis (GCA & SCA Charts)")
      try:
        lt_data = df.dropna(
            subset=[line_col, tester_col, selected_trait]
        ).copy()
        overall_mean = lt_data[selected_trait].mean()

        # GCA Lines
        line_means = lt_data.groupby(line_col)[selected_trait].mean()
        gca_lines = (line_means - overall_mean).reset_index()
        gca_lines.columns = [line_col, "GCA_Line"]
        gca_lines["Effect_Type"] = np.where(
            gca_lines["GCA_Line"] >= 0, "Positive (+)", "Negative (-)"
        )

        # GCA Testers
        tester_means = lt_data.groupby(tester_col)[selected_trait].mean()
        gca_testers = (tester_means - overall_mean).reset_index()
        gca_testers.columns = [tester_col, "GCA_Tester"]
        gca_testers["Effect_Type"] = np.where(
            gca_testers["GCA_Tester"] >= 0, "Positive (+)", "Negative (-)"
        )

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
        cross_means["Cross_Name"] = (
            cross_means[line_col].astype(str)
            + " × "
            + cross_means[tester_col].astype(str)
        )

        col1, col2 = st.columns(2)

        with col1:
          st.write("### Female Line GCA Effects (Additive)")
          fig_gca_l = px.bar(
              gca_lines.sort_values(by="GCA_Line"),
              x="GCA_Line",
              y=line_col,
              orientation="h",
              color="Effect_Type",
              color_discrete_map={
                  "Positive (+)": "#2ca02c",
                  "Negative (-)": "#d62728",
              },
              title="Line GCA Effects (Diverging from Zero)",
              text_auto=".2f",
          )
          fig_gca_l.add_vline(x=0, line_width=1.5, line_dash="dash")
          st.plotly_chart(fig_gca_l, use_container_width=True)

        with col2:
          st.write("### Male Tester GCA Effects (Additive)")
          fig_gca_t = px.bar(
              gca_testers.sort_values(by="GCA_Tester"),
              x="GCA_Tester",
              y=tester_col,
              orientation="h",
              color="Effect_Type",
              color_discrete_map={
                  "Positive (+)": "#1f77b4",
                  "Negative (-)": "#ff7f0e",
              },
              title="Tester GCA Effects (Diverging from Zero)",
              text_auto=".2f",
          )
          fig_gca_t.add_vline(x=0, line_width=1.5, line_dash="dash")
          st.plotly_chart(fig_gca_t, use_container_width=True)

        st.write("### Hybrid Specific Combining Ability (SCA) Ranking")
        top_sca = cross_means.sort_values(
            by="SCA_Cross", ascending=False
        ).head(15)
        fig_sca_bar = px.bar(
            top_sca,
            x="Cross_Name",
            y="SCA_Cross",
            color="SCA_Cross",
            color_continuous_scale="Tealrose",
            title="Top Hybrids by Specific Combining Ability (SCA)",
            text_auto=".2f",
        )
        st.plotly_chart(fig_sca_bar, use_container_width=True)

      except Exception as e:
        st.error(f"Error computing Line x Tester effects: {e}")

    # ----------------------------------------------------
    # TAB 3: GxE Reaction Norm Plots (Line Graph)
    # ----------------------------------------------------
    with tab3:
      st.subheader("Genotype × Environment Reaction Norms (Stability Plot)")
      try:
        gxe_df = (
            df.groupby([genotype_col, env_col])[selected_trait]
            .mean()
            .reset_index()
        )

        # Plot reaction norm lines across locations
        fig_gxe_line = px.line(
            gxe_df,
            x=env_col,
            y=selected_trait,
            color=genotype_col,
            markers=True,
            title=(
                f"Reaction Norm Plot: Stability of {selected_trait} Across"
                " Locations"
            ),
            labels={
                selected_trait: f"Mean {selected_trait}",
                env_col: "Location / Environment",
            },
        )
        fig_gxe_line.update_layout(hovermode="x unified")
        st.plotly_chart(fig_gxe_line, use_container_width=True)

        st.info(
            "💡 **How to Read This Diagram:** Lines that stay horizontal and high"
            " represent **stable, high-yielding genotypes**. Lines that cross over"
            " heavily show strong **Genotype × Environment (GxE) interaction**."
        )

      except Exception as e:
        st.error(f"Unable to render Reaction Norm Plot: {e}")

    # ----------------------------------------------------
    # TAB 4: Predictive Breeding (GEBV Scatter & Bar Charts)
    # ----------------------------------------------------
    with tab4:
      st.subheader("Predictive Genomic Selection & GEBV Rankings")

      if marker_cols:
        try:
          X = df[marker_cols].fillna(0)
          y = df[selected_trait].fillna(df[selected_trait].mean())

          model = Ridge(alpha=1.0)
          model.fit(X, y)
          df["Predicted_GEBV"] = model.predict(X)

          col_p1, col_p2 = st.columns(2)

          with col_p1:
            st.write("### Model Accuracy: Observed vs Predicted GEBV")
            fig_scatter = px.scatter(
                df,
                x=selected_trait,
                y="Predicted_GEBV",
                color=genotype_col,
                trendline="ols",
                title=(
                    f"Observed {selected_trait} vs. Genomic Predicted Value"
                ),
                labels={
                    selected_trait: f"Observed {selected_trait}",
                    "Predicted_GEBV": "Predicted GEBV",
                },
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

          with col_p2:
            st.write("### Top Ranked Genotypes by GEBV")
            gebv_rank = (
                df.groupby(genotype_col)["Predicted_GEBV"].mean().reset_index()
            )
            gebv_rank = gebv_rank.sort_values(
                by="Predicted_GEBV", ascending=False
            ).head(15)

            fig_gebv_bar = px.bar(
                gebv_rank,
                x="Predicted_GEBV",
                y=genotype_col,
                orientation="h",
                color="Predicted_GEBV",
                color_continuous_scale="Plasma",
                title="Top 15 Genotypes Ranked by GEBV",
                text_auto=".2f",
            )
            fig_gebv_bar.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_gebv_bar, use_container_width=True)

        except Exception as e:
          st.error(f"Error running GEBV predictions: {e}")
      else:
        st.info(
            "No SNP marker columns starting with 'M_' detected. Showing top"
            " genotypes by observed phenotypic performance."
        )
        mean_rank = (
            df.groupby(genotype_col)[selected_trait]
            .mean()
            .reset_index()
            .sort_values(by=selected_trait, ascending=False)
            .head(15)
        )
        fig_rank = px.bar(
            mean_rank,
            x=selected_trait,
            y=genotype_col,
            orientation="h",
            color=selected_trait,
            title=f"Top 15 Genotypes by Observed Mean {selected_trait}",
            text_auto=".2f",
        )
        fig_rank.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_rank, use_container_width=True)

    # ----------------------------------------------------
    # TAB 5: Export All Data
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
          label="📥 Download Complete Quantitative Genetics Excel Report (.xlsx)",
          data=buffer.getvalue(),
          file_name="predictive_breeding_analysis_report.xlsx",
          mime="application/vnd.ms-excel",
      )

  except Exception as main_err:
    st.error(f"An error occurred while loading dataset: {main_err}")

else:
  st.info(
      "👈 Upload an Excel (.xlsx) or CSV (.csv) file in the sidebar to generate"
      " diagrams and analysis."
  )
