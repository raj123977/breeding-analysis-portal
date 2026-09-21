import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

# --- Page Configuration for Cross-Device Responsiveness ---
st.set_page_config(
    page_title="Predictive Breeding Portal", page_icon="🌾", layout="wide"
)

st.title("🌾 Predictive Breeding Analysis Portal")
st.markdown(
    "Upload breeding trial Excel data to generate predictions, GEBVs, and analytics."
)

# --- File Upload Section ---
uploaded_file = st.sidebar.file_uploader(
    "Upload Breeding Data (.xlsx or .csv)", type=["xlsx", "csv"]
)


@st.cache_data
def load_data(file):
  if file.name.endswith(".xlsx"):
    return pd.read_excel(file)
  else:
    return pd.read_csv(file)


if uploaded_file is not None:
  df = load_data(uploaded_file)
  st.sidebar.success("Data successfully loaded!")

  # Variable Validation
  st.subheader("1. Variable Format Verification")
  st.write("Data Preview:", df.head(5))

  cols = df.columns.tolist()
  genotype_col = st.selectbox(
      "Select Genotype Column:",
      cols,
      index=cols.index("Genotype_ID") if "Genotype_ID" in cols else 0,
  )

  trait_cols = st.multiselect(
      "Select Target Traits for Prediction/Analysis:",
      [c for c in cols if c != genotype_col],
      default=[c for c in cols if "Trait_" in c] or [cols[1]],
  )

  marker_cols = [c for c in cols if c.startswith("M_")]

  # --- Analytics Tabs ---
  tab1, tab2, tab3, tab4, tab5 = st.tabs([
      "📊 Summary Statistics",
      "🔥 Trait Correlations",
      "🧬 Genetic PCA",
      "🔮 Predictive Breeding Models",
      "📥 Export Results",
  ])

  # Tab 1: Trait Distributions
  with tab1:
    st.subheader("Trait Distribution & Summary")
    st.dataframe(df[trait_cols].describe().T)

    selected_trait = st.selectbox("Select Trait to Visualize:", trait_cols)
    fig_hist = px.histogram(
        df,
        x=selected_trait,
        color=genotype_col if len(df[genotype_col].unique()) < 20 else None,
        marginal="box",
        title=f"Distribution of {selected_trait}",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

  # Tab 2: Trait Correlations
  with tab2:
    st.subheader("Phenotypic & Genetic Correlation Heatmap")
    if len(trait_cols) > 1:
      corr = df[trait_cols].corr()
      fig_corr = px.imshow(
          corr, text_auto=True, color_continuous_scale="Viridis"
      )
      st.plotly_chart(fig_corr, use_container_width=True)
    else:
      st.info("Select at least two traits to calculate correlations.")

  # Tab 3: PCA Analysis
  with tab3:
    st.subheader("Population Structure / Principal Component Analysis")
    pca_features = (
        marker_cols
        if len(marker_cols) > 0
        else [c for c in trait_cols if pd.api.types.is_numeric_dtype(df[c])]
    )
    if len(pca_features) >= 2:
      pca_df = df[pca_features].fillna(df[pca_features].mean())
      pca = PCA(n_components=2)
      components = pca.fit_transform(pca_df)
      pca_result = pd.DataFrame(
          components, columns=["PC1", "PC2"], index=df.index
      )
      pca_result[genotype_col] = df[genotype_col]

      fig_pca = px.scatter(
          pca_result,
          x="PC1",
          y="PC2",
          hover_data=[genotype_col],
          title="PCA Scatter Plot (Genetic Clustering)",
      )
      st.plotly_chart(fig_pca, use_container_width=True)
    else:
      st.warning(
          "Insufficient numeric marker or trait columns for PCA calculation."
      )

  # Tab 4: Predictive Modeling (GEBV / Breeding Values)
  with tab4:
    st.subheader("Predictive Breeding Value Estimates (GEBV)")
    target_predict_trait = st.selectbox(
        "Select Target Trait to Predict:", trait_cols, key="pred_trait"
    )

    if marker_cols:
      st.info(f"Using {len(marker_cols)} SNP Markers for Ridge Regression (GBLUP proxy).")
      X = df[marker_cols].fillna(0)
      y = df[target_predict_trait].fillna(df[target_predict_trait].mean())

      model = Ridge(alpha=1.0)
      model.fit(X, y)
      df["Predicted_GEBV"] = model.predict(X)

      fig_gebv = px.bar(
          df.sort_values(by="Predicted_GEBV", ascending=False),
          x=genotype_col,
          y="Predicted_GEBV",
          title=f"Ranked Genomic Estimated Breeding Values ({target_predict_trait})",
      )
      st.plotly_chart(fig_gebv, use_container_width=True)
    else:
      # Phenotypic Index Prediction if Markers are absent
      st.info(
          "No 'M_' marker columns detected. Calculating selection index based"
          " on standardized traits."
      )
      df["Selection_Index"] = (
          df[trait_cols]
          .apply(lambda x: (x - x.mean()) / x.std())
          .sum(axis=1)
      )
      fig_index = px.bar(
          df.sort_values(by="Selection_Index", ascending=False),
          x=genotype_col,
          y="Selection_Index",
          title="Genotype Ranking by Multi-Trait Selection Index",
      )
      st.plotly_chart(fig_index, use_container_width=True)

  # Tab 5: Data Export
  with tab5:
    st.subheader("Save and Download Analyzed Results")

    # CSV Download
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📄 Download Analysis Data (CSV)",
        data=csv_data,
        file_name="predictive_breeding_analysis.csv",
        mime="text/csv",
    )

    # Excel Download
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
      df.to_excel(writer, sheet_name="Predictive_Results", index=False)
    st.download_button(
        label="📊 Download Analysis Data (Excel .xlsx)",
        data=buffer.getvalue(),
        file_name="predictive_breeding_analysis.xlsx",
        mime="application/vnd.ms-excel",
    )

else:
  st.info("👈 Please upload an Excel or CSV file in the sidebar to begin analysis.")