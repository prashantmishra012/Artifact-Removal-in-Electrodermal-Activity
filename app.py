import streamlit as st
import pandas as pd
import numpy as np
import sys
import io

# Add src to path to import custom modules
sys.path.append('src')

from data.preprocess import preprocess_eda
from visualization.plot import plot_eda_comparison
from pipelines.inference_pipeline import clean_signal_with_autoencoder
from models.autoencoder import build_autoencoder  # Required for TF to load custom model components if any

st.set_page_config(
    page_title="EDA Artifact Removal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Unsupervised Artifact Removal in Electrodermal Activity (EDA) Signals")

st.markdown("""
Welcome to the EDA Signal Cleaner. This tool uses a **Convolutional Autoencoder** to automatically detect and remove
motion artifacts from raw EDA data, helping you get a cleaner signal for analysis.
""")

st.sidebar.header("Settings & Parameters")
segment_length = st.sidebar.slider(
    "Segment Length for Autoencoder (samples)",
    min_value=100,
    max_value=2000,
    value=700,
    step=50,
    help="Segment length should match the model training segment size for best results.",
)

st.sidebar.markdown("""
---
**Developed by the JOVAC Project**

*Upload a raw EDA signal CSV file to begin.*
""")

uploaded_file = st.file_uploader("📂 Choose a raw EDA signal CSV file", type="csv")

def convert_df_to_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')

if uploaded_file is not None:
    try:
        # Read raw data
        raw_eda_df = pd.read_csv(uploaded_file)
        raw_eda = raw_eda_df.iloc[:, 0].to_numpy()

        st.header("Signal Cleaning Results")

        with st.spinner("🤖 The autoencoder is analyzing and cleaning your signal..."):
            # Ensure model file exists in 'src/models/autoencoder.h5'
            model_path = 'src/models/autoencoder.h5'

            # Clean raw signal
            cleaned_eda = clean_signal_with_autoencoder(raw_eda, model_path, segment_length=segment_length)

        st.success("✅ Signal cleaned successfully!")

        # --- Display results in columns ---
        col1, col2 = st.columns((2, 1))

        with col1:
            # Visualization
            st.subheader("Raw vs. Cleaned Signal")
            fig = plot_eda_comparison(raw_eda, {'Autoencoder Cleaned': cleaned_eda},
                                      title="Raw vs. Autoencoder Cleaned Signal")
            st.pyplot(fig)

        with col2:
            st.subheader("Cleaned Data")
            # Provide download option for cleaned signal
            cleaned_df = pd.DataFrame({'Cleaned_EDA': cleaned_eda})
            csv_bytes = convert_df_to_csv_bytes(cleaned_df)

            st.download_button(
                label="📥 Download Cleaned EDA CSV",
                data=csv_bytes,
                file_name='cleaned_eda.csv',
                mime='text/csv',
                help="Download the cleaned EDA signal as a CSV file.",
                use_container_width=True
            )
            st.dataframe(cleaned_df, use_container_width=True)

        with st.expander("NeuroKit2 Preprocessing Details"):
            preprocessed_df = preprocess_eda(raw_eda)
            st.dataframe(preprocessed_df)

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")

else:
    st.markdown("""
    #### Instructions
    1.  **Upload a CSV file** containing a single column of raw EDA signal values.
    2.  The autoencoder will automatically denoise the signal and display a comparison plot.
    3.  Use the sidebar to adjust the **Segment Length** parameter if needed (must match the model's training).
    4.  **Download the cleaned signal** for your own analysis.
    """)
