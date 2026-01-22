import streamlit as st
import pickle
import pandas as pd
import plotly.express as px

# ---------------------------------
# Page config (must be FIRST)
# ---------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------
# Custom CSS for styling
# ---------------------------------
st.markdown("""
    <style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header styling */
    h1 {
        text-align: center;
        color: #1f77b4;
        padding-bottom: 1rem;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 2rem;
    }
    
    /* Subheader styling */
    h2, h3 {
        color: #2c3e50;
        margin-top: 1.5rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        padding-top: 1rem;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #1f77b4 0%, #2c3e50 100%);
        color: white;
        font-weight: bold;
        font-size: 1.1rem;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(31, 119, 180, 0.4);
    }
    
    /* Metric cards styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1rem;
        font-weight: 600;
    }
    
    /* Container styling */
    .prediction-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-top: 1rem;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
    
    /* Plot container styling */
    .plot-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Ensure Plotly charts are visible */
    .js-plotly-plot {
        width: 100% !important;
        height: auto !important;
        min-height: 600px !important;
    }
    
    .plotly {
        width: 100% !important;
        height: 100% !important;
    }
    
    /* Column styling for better plot visibility */
    [data-testid="column"] {
        min-width: 0;
    }
    
    /* Ensure plotly container has proper dimensions */
    div[data-testid="stPlotlyChart"] {
        width: 100%;
        min-height: 700px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------
# Cache model & scaler (performance)
# ---------------------------------
@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_scaler():
    with open("scaler.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
scaler = load_scaler()

# ---------------------------------
# Feature definitions
# ---------------------------------
feature_names = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "EstimatedSalary", "Geography_France", "Geography_Germany", "Geography_Spain",
    "Gender_Female", "Gender_Male", "HasCrCard_0", "HasCrCard_1",
    "IsActiveMember_0", "IsActiveMember_1"
]

scale_vars = ["CreditScore", "EstimatedSalary", "Tenure", "Balance", "Age", "NumOfProducts"]

default_values = [
    600, 30, 2, 8000, 2, 60000,
    True, False, False, True, False, False, True, False, True
]

# ---------------------------------
# Sidebar
# ---------------------------------
st.sidebar.image("media/Pic 1.PNG", use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.header("📊 User Inputs")
st.sidebar.markdown("---")

user_inputs = {}

for i, feature in enumerate(feature_names):
    if feature in scale_vars:
        user_inputs[feature] = st.sidebar.number_input(
            feature, value=float(default_values[i]), step=1.0, key=f"input_{i}"
        )
    else:
        user_inputs[feature] = st.sidebar.checkbox(
            feature, value=default_values[i], key=f"checkbox_{i}"
        )

# Convert to DataFrame
input_data = pd.DataFrame([user_inputs])

# Scale required columns
input_scaled = input_data.copy()
input_scaled[scale_vars] = scaler.transform(input_data[scale_vars])

# ---------------------------------
# Main Header
# ---------------------------------
st.image("media/Pic 2.PNG", use_container_width=True)
st.markdown("<h1>Customer Churn Prediction System</h1>", unsafe_allow_html=True)

# Add some spacing
st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------
# Layout - More responsive columns
# ---------------------------------
left_col, right_col = st.columns([1.5, 1], gap="large")

# ---------------------------------
# Feature Importance Section
# ---------------------------------
with left_col:
    st.markdown("### 📈 Feature Importance Analysis")
    st.markdown("---")

    try:
        feature_importance_df = pd.read_excel(
            "feature_importance.xlsx",
            usecols=["Feature", "Feature Importance Score"]
        )

        # Sort data and create plot (using old graph reference style)
        feature_importance_sorted = feature_importance_df.sort_values(
            by="Feature Importance Score", ascending=False
        )
        
        # Create the feature importance bar chart (old style with larger dimensions)
        fig = px.bar(
            feature_importance_sorted,
            x="Feature Importance Score",
            y="Feature",
            orientation="h",
            title="Feature Importance",
            labels={"Feature Importance Score": "Importance", "Feature": "Features"},
            width=800,  # Larger width for better visibility
            height=700  # Larger height for better visibility
        )
        
        # Display the plot
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading feature importance data: {str(e)}")
        st.info("Please ensure 'feature_importance.xlsx' file exists in the current directory.")

# ---------------------------------
# Prediction Section
# ---------------------------------
with right_col:
    st.markdown("### 🎯 Churn Prediction")
    st.markdown("---")

    # Prediction button with better styling
    if st.button("🔮 Predict Churn", use_container_width=True, type="primary"):
        probs = model.predict_proba(input_scaled)[0]
        prediction = model.predict(input_scaled)[0]

        label = "Churned ❌" if prediction == 1 else "Retain ✅"
        label_color = "#e74c3c" if prediction == 1 else "#27ae60"

        # Prediction result container
        st.markdown(f"""
        <div class="prediction-container">
            <h3 style="text-align: center; color: {label_color}; margin-bottom: 1.5rem;">
                {label}
            </h3>
        </div>
        """, unsafe_allow_html=True)

        # Metrics with better spacing
        col1, col2 = st.columns(2, gap="medium")
        
        with col1:
            st.metric(
                "Churn Probability", 
                f"{probs[1]*100:.2f}%",
                delta=f"{probs[1]*100:.2f}%",
                delta_color="inverse"
            )
        
        with col2:
            st.metric(
                "Retention Probability", 
                f"{probs[0]*100:.2f}%",
                delta=f"{probs[0]*100:.2f}%",
                delta_color="normal"
            )
        
        # Progress bars for visual representation
        st.markdown("---")
        st.markdown("**Probability Distribution:**")
        
        # Convert numpy float32 to Python float for st.progress()
        retention_prob = float(probs[0])
        churn_prob = float(probs[1])
        
        st.progress(retention_prob, text=f"Retention: {retention_prob*100:.1f}%")
        st.progress(churn_prob, text=f"Churn: {churn_prob*100:.1f}%")
    else:
        st.info("👆 Click the button above to predict customer churn based on the input features.")

# Streamlit run churn_predict_streamlit_app.py
