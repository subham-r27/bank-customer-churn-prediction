# Bank Customer Churn Prediction 🏦

A comprehensive machine learning project for predicting bank customer churn, featuring data analysis, model development, interactive web deployment, and business intelligence dashboards.

## 📋 Overview


https://github.com/user-attachments/assets/dcf2cad7-e0f8-4d30-8ffb-f50d18a5c1b6



This project provides an end-to-end solution for predicting customer churn in the banking sector. It combines machine learning model development, interactive web application deployment, and business intelligence visualization to help banks identify at-risk customers and take proactive retention measures.

## 🎯 Project Components

### 1. 🤖 Machine Learning Model & Analysis
- **XGBoost Classifier**: Advanced gradient boosting model for churn prediction
- **Data Preprocessing**: Comprehensive data cleaning, feature engineering, and scaling
- **Hyperparameter Tuning**: Optimized model performance through systematic parameter tuning
- **Feature Importance Analysis**: Identification of key factors driving customer churn
- **Model Evaluation**: Performance metrics and validation analysis

### 2. 🌐 Streamlit Web Application
- **Interactive Prediction Interface**: Real-time churn prediction with user-friendly inputs
- **Feature Importance Visualization**: Interactive charts showing model insights
- **Probability Scores**: Detailed churn and retention probability metrics
- **Responsive Design**: Modern, intuitive UI for seamless user experience

### 3. 📊 PowerBI Analysis Dashboard
- **Business Intelligence Visualization**: Comprehensive dashboards for stakeholder insights
- **Churn Analytics**: Visual analysis of churn patterns and trends
- **Customer Segmentation**: Interactive exploration of customer demographics
- **Performance Metrics**: Key performance indicators and business metrics

## ✨ Key Features

- **Predictive Analytics**: Identify customers at risk of churning with high accuracy
- **Real-time Predictions**: Instant churn probability calculations
- **Feature Insights**: Understand which factors most influence churn decisions
- **Interactive Dashboards**: Visual exploration of customer data and model performance
- **Production-Ready**: Deployed web application for immediate use
- **Business Intelligence**: Comprehensive analytics for strategic decision-making

## 📊 Dataset

The project uses a bank customer dataset containing:
- **Demographic Features**: Age, Geography, Gender
- **Financial Features**: Credit Score, Balance, Estimated Salary
- **Behavioral Features**: Tenure, Number of Products, Active Membership
- **Service Features**: Credit Card Status
- **Target Variable**: Churn Status (Exited/Retained)

## 🛠️ Technologies Used

### Machine Learning & Data Science
- **XGBoost**: Gradient boosting framework for classification
- **Scikit-learn**: Machine learning utilities and preprocessing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing

### Web Application
- **Streamlit**: Interactive web application framework
- **Plotly**: Interactive data visualization
- **Pickle**: Model serialization and loading

### Business Intelligence
- **PowerBI**: Data visualization and business intelligence platform
- **Excel/CSV**: Data export and integration

### Development Tools
- **Jupyter Notebook**: Interactive data analysis and model development
- **Python 3.7+**: Programming language

## 📦 Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)
- PowerBI Desktop (for dashboard analysis)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd "Bank Customer Churn Prediction"
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install packages individually:

```bash
pip install streamlit pandas plotly openpyxl scikit-learn xgboost numpy
```

## 🚀 Usage

### Running the Streamlit Application

1. **Ensure Required Files Are Present**:
   - `churn_predict_streamlit_app.py` - Main application file
   - `best_model.pkl` - Trained ML model
   - `scaler.pkl` - Feature scaler
   - `feature_importance.xlsx` - Feature importance data
   - `media/` folder with required images

2. **Launch the Application**:
   ```bash
   streamlit run churn_predict_streamlit_app.py
   ```

3. **Access the Application**:
   - The app will automatically open in your browser at `http://localhost:8501`
   - Use the sidebar to input customer features
   - Click "🔮 Predict Churn" to get predictions

### Using the Machine Learning Model

1. **Open the Jupyter Notebook**:
   ```bash
   jupyter notebook Predicting_churn.ipynb
   ```

2. **Run the Analysis**:
   - Execute cells sequentially to perform data preprocessing
   - Train and evaluate the XGBoost model
   - Generate feature importance analysis
   - Export model artifacts for deployment

### PowerBI Dashboard

1. **Open PowerBI Desktop**
2. **Import the Dataset**: Load `Churn_Modelling.csv` or `bank_churn_data.xlsx`
3. **Create Visualizations**:
   - Churn rate by demographics
   - Customer distribution analysis
   - Feature correlation analysis
   - Predictive insights visualization
4. **Publish Dashboard**: Share insights with stakeholders

## 📁 Project Structure

```
Bank Customer Churn Prediction/
│
├── churn_predict_streamlit_app.py    # Streamlit web application
├── Predicting_churn.ipynb            # Jupyter notebook for ML analysis
├── best_model.pkl                     # Trained XGBoost model
├── scaler.pkl                         # Feature scaler for preprocessing
├── feature_importance.xlsx           # Feature importance analysis
├── Churn_Modelling.csv               # Original dataset
├── bank_churn_data.xlsx              # Alternative data format
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
│
└── media/                            # Media files directory
    ├── Pic 1.png                     # Sidebar image
    ├── Pic 2.png                     # Header image
    ├── ML Process.PNG                # ML process visualization
    └── project_flow.png              # Project workflow diagram
```

## 🔍 Model Details

### Algorithm
- **XGBoost Classifier**: Extreme Gradient Boosting for binary classification

### Features
- Credit Score
- Age
- Tenure (years with bank)
- Balance
- Number of Products
- Estimated Salary
- Geography (France, Germany, Spain)
- Gender
- Has Credit Card
- Is Active Member

### Preprocessing
- Feature scaling using StandardScaler
- One-hot encoding for categorical variables
- Data normalization

## 💡 Usage Examples

### Streamlit Application

1. **Input Customer Information**:
   - Enter customer demographics and financial details in the sidebar
   - Adjust feature values using sliders and checkboxes

2. **View Predictions**:
   - Click "🔮 Predict Churn" to get instant predictions
   - Review churn probability and retention probability
   - Analyze feature importance chart

3. **Interpret Results**:
   - **Churned ❌**: Customer likely to leave (probability > 50%)
   - **Retain ✅**: Customer likely to stay (probability > 50%)

### PowerBI Dashboard

- **Churn Analysis**: Visualize churn rates across different customer segments
- **Demographic Insights**: Explore churn patterns by geography, age, gender
- **Feature Correlation**: Understand relationships between features and churn
- **Predictive Insights**: View model predictions and actual outcomes

## 📈 Model Performance

The XGBoost model has been optimized through:
- Hyperparameter tuning
- Cross-validation
- Feature importance analysis
- Performance metric evaluation

## 🔧 Troubleshooting

### Common Issues

1. **ModuleNotFoundError**:
   ```bash
   pip install -r requirements.txt
   ```

2. **FileNotFoundError for model files**:
   - Ensure `best_model.pkl` and `scaler.pkl` are in the project root

3. **Streamlit port already in use**:
   ```bash
   streamlit run churn_predict_streamlit_app.py --server.port 8502
   ```

4. **PowerBI data connection issues**:
   - Verify CSV/Excel file paths
   - Check data format compatibility

## 📝 Notes

- The model uses pre-trained artifacts (`best_model.pkl`, `scaler.pkl`)
- Feature scaling is automatically applied in the Streamlit app
- All categorical features are one-hot encoded
- PowerBI dashboard can be customized based on business requirements

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available for educational and commercial use.

## 👤 Author

Developed as a comprehensive solution for bank customer churn prediction and analysis.

## 🙏 Acknowledgments

- Dataset: Bank Customer Churn Dataset
- Libraries: XGBoost, Streamlit, PowerBI, Scikit-learn

---

**Happy Predicting! 🎯**

For questions or issues, please open an issue in the repository.
