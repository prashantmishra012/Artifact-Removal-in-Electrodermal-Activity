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

# --- Page Configuration ---
st.set_page_config(
    page_title="EDA Signal Cleaner | JOVAC",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for Modern UI ---
def load_css():
    """Inject custom CSS for styling the app."""
    st.markdown("""
        <style>
            /* --- Main Theme & Layout --- */
            body {
                color: #EAEAEA;
                background-color: #0E1117;
            }
            .stApp {
                background-color: #0E1117;
            }
            .main .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
                padding-left: 5rem;
                padding-right: 5rem;
            }

            /* --- Sidebar --- */
            .st-emotion-cache-16txtl3 {
                padding: 2rem 1rem;
            }

            /* --- Styled Containers & Cards --- */
            .styled-container {
                border: 1px solid rgba(255, 127, 80, 0.3);
                background-color: #161A25;
                padding: 2rem;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                margin-bottom: 2rem;
            }

            /* --- File Uploader --- */
            .stFileUploader {
                border: 2px dashed #FF7F50;
                background-color: rgba(255, 127, 80, 0.05);
                border-radius: 10px;
                padding: 1.5rem;
            }
            .stFileUploader label {
                font-size: 1.2rem;
                font-weight: bold;
                color: #FF7F50;
            }

            /* --- Buttons --- */
            .stButton>button {
                border-radius: 8px;
                border: 1px solid #FF7F50;
                color: #FF7F50;
                background-color: transparent;
                transition: all 0.3s ease-in-out;
            }
            .stButton>button:hover {
                background-color: #FF7F50;
                color: white;
                border-color: #FF7F50;
            }
            .stDownloadButton>button {
                width: 100%;
                background-color: #FF7F50;
                color: white;
                border: none;
                border-radius: 8px;
            }
            .stDownloadButton>button:hover {
                background-color: #E57245;
            }

            /* --- Tabs --- */
            .stTabs [data-baseweb="tab-list"] {
                gap: 24px;
            }
            .stTabs [data-baseweb="tab"] {
                height: 50px;
                white-space: pre-wrap;
                background-color: transparent;
                border-radius: 4px 4px 0px 0px;
                gap: 1px;
                padding-top: 10px;
                padding-bottom: 10px;
            }
            .stTabs [aria-selected="true"] {
                background-color: #161A25;
                color: #FF7F50;
                font-weight: bold;
            }

            /* --- Footer --- */
            footer {
                visibility: hidden;
            }
            .footer {
                position: fixed;
                left: 0;
                bottom: 0;
                width: 100%;
                background-color: #0E1117;
                color: #888;
                text-align: center;
                padding: 10px;
                font-size: 0.8rem;
                border-top: 1px solid #1E222B;
            }
        </style>
    """, unsafe_allow_html=True)

load_css()

# --- Sidebar Content ---
with st.sidebar:
    st.header("⚙️ Settings & Parameters")
    segment_length = st.slider(
        "Segment Length (samples)",
        min_value=100,
        max_value=2000,
        value=700,
        step=50,
        help="Adjust the segment length for the autoencoder. This should match the model's training for best results.",
    )

    st.markdown("---")
    st.subheader("💡 Quick Tips")
    st.info("""
    - **Upload Format:** Ensure your CSV has a single column of raw EDA data.
    - **Segment Length:** A value of 700 is optimal for the default model.
    - **Interpretation:** The cleaned signal removes motion artifacts, revealing clearer physiological responses.
    """)

    st.markdown("---")
# --- Main Application ---

# --- Hero Header ---
st.markdown("""
    <div class="styled-container" style="text-align: center;">
        <h1>⚡ Unsupervised Artifact Removal in EDA Signals</h1>
        <p>
            Using a Convolutional Autoencoder to automatically detect and remove motion artifacts from electrodermal activity (EDA) data.
        </p>
    </div>
""", unsafe_allow_html=True)

# --- Layout Columns ---
col1, col2 = st.columns((2.5, 1.5))

with col1:
    # --- File Uploader Section ---
    st.markdown('<div class="styled-container">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "📂 Upload a raw EDA signal CSV file to begin",
        type="csv",
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # --- Pipeline Overview Card ---
    st.markdown("""
        <div class="styled-container">
            <h4>Pipeline Overview</h4>
            <ol>
                <li><strong>Upload Data:</strong> Start by uploading your raw EDA signal.</li>
                <li><strong>Segmentation:</strong> The signal is divided into smaller segments.</li>
                <li><strong>Autoencoder Cleaning:</strong> Our pre-trained model reconstructs each segment, filtering out artifacts.</li>
                <li><strong>Stitching:</strong> The clean segments are reassembled into a final signal.</li>
                <li><strong>Analyze & Download:</strong> View the results and download the clean data.</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)

# --- Backend Logic & Results Display ---
def convert_df_to_csv_bytes(df):
    """Utility to convert DataFrame to CSV bytes for download."""
    return df.to_csv(index=False).encode('utf-8')

if uploaded_file is not None:
    try:
        # --- MODEL LOGIC (UNCHANGED) ---
        raw_eda_df = pd.read_csv(uploaded_file)
        raw_eda = raw_eda_df.iloc[:, 0].to_numpy()

        with st.spinner("🤖 The autoencoder is analyzing and cleaning your signal... Please wait."):
            model_path = 'src/models/autoencoder_v2.h5' # <-- UPDATE THIS LINE
            cleaned_eda = clean_signal_with_autoencoder(raw_eda, model_path, segment_length=segment_length)
            preprocessed_df = preprocess_eda(raw_eda)

        st.success("✅ Signal cleaned successfully!")
        st.markdown("---")

        # --- RESULTS SECTION (WITH TABS) ---
        tab1, tab2, tab3 = st.tabs([
            "📊 Raw vs. Cleaned Signal",
            "📑 Preprocessing Details",
            "📥 Download Data"
        ])

        with tab1:
            st.subheader("Signal Comparison Plot")
            # Use the custom, enhanced plotting function
            fig = plot_eda_comparison(
                raw_eda,
                {'Autoencoder Cleaned': cleaned_eda},
                title="Raw vs. Autoencoder Cleaned Signal"
            )
            st.pyplot(fig, use_container_width=True)

        with tab2:
            st.subheader("NeuroKit2 Preprocessing Analysis")
            st.markdown("This table shows features extracted from the raw signal using NeuroKit2, including tonic and phasic components, and detected skin conductance responses (SCRs).")
            st.dataframe(preprocessed_df, use_container_width=True)

        with tab3:
            st.subheader("Download Your Cleaned EDA Signal")
            st.markdown("The button below allows you to download the denoised EDA signal as a CSV file, ready for further analysis.")

            cleaned_df = pd.DataFrame({'Cleaned_EDA': cleaned_eda})
            csv_bytes = convert_df_to_csv_bytes(cleaned_df)

            st.download_button(
                label="Download Cleaned EDA (CSV)",
                data=csv_bytes,
                file_name=f'cleaned_eda_{uploaded_file.name}',
                mime='text/csv',
                help="Click to download the cleaned data.",
                use_container_width=True
            )
            st.markdown("---")
            st.subheader("Data Preview")
            st.dataframe(cleaned_df.head(), use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
        st.warning("Please ensure your CSV file contains a single column of numerical EDA data.")

else:
    # --- Initial Instructions (when no file is uploaded) ---
    st.info("Upload a file above to see the magic happen!")

# --- Footer ---
st.markdown("""
    <div class="footer">
        Developed by Prashant Mishra
    </div>
""", unsafe_allow_html=True)
