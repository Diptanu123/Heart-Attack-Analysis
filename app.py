import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Heart Attack Prediction App",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #e74c3c;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .risk-high {
        background: linear-gradient(90deg, #ff6b6b 0%, #ee5a52 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .risk-low {
        background: linear-gradient(90deg, #51cf66 0%, #40c057 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model and scaler
@st.cache_resource
def load_model():
    try:
        model = joblib.load(r'heart_attack_model.pkl')
        scaler = joblib.load(r'heart_attack_scaler.pkl')
        feature_names = joblib.load(r'feature_names.pkl')
        return model, scaler, feature_names
    except FileNotFoundError:
        st.error("❌ Model files not found! Please run the training script first.")
        return None, None, None

def get_user_input():
    """Create input widgets for user data"""
    
    st.sidebar.markdown("## 👤 Patient Information")
    
    # Basic Information
    age = st.sidebar.slider("Age", 20, 100, 50)
    sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
    sex_val = 1 if sex == "Male" else 0
    
    # Chest Pain Type
    st.sidebar.markdown("### 💔 Chest Pain Information")
    chest_pain_options = {
        "Typical Angina": 1,
        "Atypical Angina": 2, 
        "Non-anginal Pain": 3,
        "Asymptomatic": 4
    }
    chest_pain = st.sidebar.selectbox("Chest Pain Type", list(chest_pain_options.keys()))
    chest_pain_val = chest_pain_options[chest_pain]
    
    # Vital Signs
    st.sidebar.markdown("### 🩺 Vital Signs")
    resting_bp = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.sidebar.slider("Cholesterol Level (mg/dl)", 100, 600, 250)
    
    # Blood Sugar
    fasting_bs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
    fasting_bs_val = 1 if fasting_bs == "Yes" else 0
    
    # ECG Results
    st.sidebar.markdown("### 📈 Test Results")
    ecg_options = {
        "Normal": 0,
        "ST-T Wave Abnormality": 1,
        "Left Ventricular Hypertrophy": 2
    }
    resting_ecg = st.sidebar.selectbox("Resting ECG", list(ecg_options.keys()))
    resting_ecg_val = ecg_options[resting_ecg]
    
    # Exercise Information
    st.sidebar.markdown("### 🏃‍♂️ Exercise Test")
    max_hr = st.sidebar.slider("Maximum Heart Rate Achieved", 60, 220, 150)
    exercise_angina = st.sidebar.selectbox("Exercise Induced Angina", ["No", "Yes"])
    exercise_angina_val = 1 if exercise_angina == "Yes" else 0
    
    # ST Depression
    oldpeak = st.sidebar.slider("ST Depression (Oldpeak)", 0.0, 10.0, 1.0, 0.1)
    
    # ST Slope
    slope_options = {
        "Upsloping": 1,
        "Flat": 2,
        "Downsloping": 3
    }
    st_slope = st.sidebar.selectbox("ST Slope", list(slope_options.keys()))
    st_slope_val = slope_options[st_slope]
    
    # Create feature array
    features = np.array([[
        age, sex_val, chest_pain_val, resting_bp, cholesterol, 
        fasting_bs_val, resting_ecg_val, max_hr, exercise_angina_val, 
        oldpeak, st_slope_val
    ]])
    
    # Create feature names for display
    feature_dict = {
        'Age': age,
        'Sex': sex,
        'Chest Pain Type': chest_pain,
        'Resting BP': f"{resting_bp} mm Hg",
        'Cholesterol': f"{cholesterol} mg/dl",
        'Fasting Blood Sugar': fasting_bs,
        'Resting ECG': resting_ecg,
        'Max Heart Rate': max_hr,
        'Exercise Angina': exercise_angina,
        'ST Depression': oldpeak,
        'ST Slope': st_slope
    }
    
    return features, feature_dict

def create_risk_visualization(probability):
    """Create a risk gauge visualization"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Heart Attack Risk (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': "lightgreen"},
                {'range': [25, 50], 'color': "yellow"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_feature_importance_chart():
    """Create a feature importance chart based on typical importance"""
    features = ['Chest Pain Type', 'Max Heart Rate', 'ST Depression', 'Age', 'Exercise Angina', 
               'ST Slope', 'Cholesterol', 'Resting BP', 'Sex', 'Resting ECG', 'Fasting Blood Sugar']
    importance = [0.18, 0.15, 0.14, 0.12, 0.11, 0.10, 0.08, 0.06, 0.04, 0.03, 0.02]
    
    fig = px.bar(
        x=importance,
        y=features,
        orientation='h',
        title="Feature Importance in Heart Attack Prediction",
        labels={'x': 'Importance Score', 'y': 'Features'},
        color=importance,
        color_continuous_scale='Reds'
    )
    fig.update_layout(height=400)
    return fig

def display_patient_summary(feature_dict):
    """Display patient information in a nice format"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👤 Basic Information")
        st.write(f"**Age:** {feature_dict['Age']} years")
        st.write(f"**Sex:** {feature_dict['Sex']}")
        st.write(f"**Chest Pain:** {feature_dict['Chest Pain Type']}")
        
        st.markdown("### 🩺 Vital Signs")
        st.write(f"**Resting BP:** {feature_dict['Resting BP']}")
        st.write(f"**Cholesterol:** {feature_dict['Cholesterol']}")
        st.write(f"**Max Heart Rate:** {feature_dict['Max Heart Rate']}")
    
    with col2:
        st.markdown("### 📈 Test Results")
        st.write(f"**Fasting Blood Sugar:** {feature_dict['Fasting Blood Sugar']}")
        st.write(f"**Resting ECG:** {feature_dict['Resting ECG']}")
        st.write(f"**Exercise Angina:** {feature_dict['Exercise Angina']}")
        st.write(f"**ST Depression:** {feature_dict['ST Depression']}")
        st.write(f"**ST Slope:** {feature_dict['ST Slope']}")

def main():
    # Main title
    st.markdown("<h1 class='main-header'>❤️ Heart Attack Prediction App</h1>", unsafe_allow_html=True)
    
    # Load model
    model, scaler, feature_names = load_model()
    
    if model is None:
        st.stop()
    
    # Sidebar for inputs
    st.sidebar.title("🔬 Enter Patient Data")
    st.sidebar.markdown("Please fill in the patient information below:")
    
    # Get user input
    user_features, feature_dict = get_user_input()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 📊 Patient Summary")
        display_patient_summary(feature_dict)
        
        # Prediction button
        if st.button("🔍 Predict Heart Attack Risk", key="predict_btn"):
            # Scale the features
            user_features_scaled = scaler.transform(user_features)
            
            # Make prediction
            prediction = model.predict(user_features_scaled)[0]
            probability = model.predict_proba(user_features_scaled)[0]
            
            # Display results
            st.markdown("## 🎯 Prediction Results")
            
            col_result1, col_result2 = st.columns(2)
            
            with col_result1:
                if prediction == 1:
                    st.markdown("""
                    <div class='risk-high'>
                        <h3>⚠️ HIGH RISK</h3>
                        <p>Patient shows high risk of heart attack</p>
                        <p><strong>Recommendation:</strong> Immediate medical consultation required</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class='risk-low'>
                        <h3>✅ LOW RISK</h3>
                        <p>Patient shows low risk of heart attack</p>
                        <p><strong>Recommendation:</strong> Maintain healthy lifestyle</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col_result2:
                st.markdown(f"""
                <div class='metric-card'>
                    <h3>Risk Probability</h3>
                    <h2>{probability[1]:.2%}</h2>
                    <p>Confidence: {max(probability):.2%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Risk gauge
            st.plotly_chart(create_risk_visualization(probability[1]), use_container_width=True)
            
            # Additional insights
            st.markdown("## 💡 Key Insights")
            
            if probability[1] > 0.7:
                st.error("🚨 **Critical Risk Factors Detected:**")
                if feature_dict['Chest Pain Type'] in ['Typical Angina', 'Atypical Angina']:
                    st.write("- Chest pain type indicates possible cardiac issues")
                if feature_dict['Exercise Angina'] == 'Yes':
                    st.write("- Exercise-induced angina is a significant risk factor")
                if int(feature_dict['Max Heart Rate']) < 100:
                    st.write("- Low maximum heart rate may indicate heart problems")
            elif probability[1] > 0.4:
                st.warning("⚠️ **Moderate Risk - Monitor These Factors:**")
                st.write("- Regular health check-ups recommended")
                st.write("- Monitor blood pressure and cholesterol levels")
                st.write("- Maintain regular exercise routine")
            else:
                st.success("✅ **Low Risk - Maintain Current Health Status:**")
                st.write("- Continue healthy lifestyle habits")
                st.write("- Regular preventive check-ups")
                st.write("- Keep monitoring key health indicators")
    
    with col2:
        st.markdown("## 📈 Model Information")
        st.info("""
        **ExtraTreesClassifier Model**
        - Accuracy: 90%
        - Features: 11 clinical parameters
        - Training Data: 918 patients
        - Cross-validated performance
        """)
        
        # Feature importance chart
        st.plotly_chart(create_feature_importance_chart(), use_container_width=True)
        
        # Health tips
        st.markdown("## 💪 Health Tips")
        st.markdown("""
        **Prevention is Key:**
        - 🏃‍♂️ Regular exercise (30 min/day)
        - 🥗 Healthy diet (low saturated fat)
        - 🚭 Avoid smoking
        - 🍷 Limit alcohol consumption
        - 😌 Manage stress levels
        - 💊 Take prescribed medications
        - 🩺 Regular health check-ups
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only. 
        Always consult with healthcare professionals for medical decisions.</p>
        <p>Developed with ❤️ using Streamlit | Model: ExtraTreesClassifier (90% accuracy)</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":

    main()
