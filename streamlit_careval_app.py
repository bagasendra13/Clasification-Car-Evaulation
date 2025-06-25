import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Car Evaluation Prediction App",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

# Load model components
@st.cache_resource
def load_model():
    try:
        components = joblib.load('car_evaluation_prediction_components.joblib')
        return components
    except FileNotFoundError:
        st.error("Model file 'car_evaluation_prediction_components.joblib' not found!")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

# Mapping dictionaries
buying_map = {'vhigh': 4, 'high': 3, 'med': 2, 'low': 1}
maint_map = {'vhigh': 4, 'high': 3, 'med': 2, 'low': 1}
doors_map = {'2': 2, '3': 3, '4': 4, '5more': 5}
persons_map = {'2': 2, '4': 4, 'more': 5}
lug_boot_map = {'small': 1, 'med': 2, 'big': 3}
safety_map = {'low': 1, 'med': 2, 'high': 3}
class_map = {'unacc': 1, 'acc': 2, 'good': 3, 'vgood': 4}

# Reverse class map for display
class_map_reverse = {v: k for k, v in class_map.items()}

def predict_car(data, model_components):
    df = pd.DataFrame([data])

    # Apply mapping
    df['buying'] = df['buying'].map(buying_map)
    df['maint'] = df['maint'].map(maint_map)
    df['doors'] = df['doors'].map(doors_map)
    df['persons'] = df['persons'].map(persons_map)
    df['lug_boot'] = df['lug_boot'].map(lug_boot_map)
    df['safety'] = df['safety'].map(safety_map)

    model = model_components['model']
    feature_names = model_components['feature_names']
    df_for_pred = df[feature_names].copy()

    prediction = model.predict(df_for_pred)[0]
    probabilities = model.predict_proba(df_for_pred)[0]

    return {
        'prediction': int(prediction),
        'prediction_label': class_map_reverse[prediction],
        'probability': float(probabilities[prediction - 1]),
        'probabilities': probabilities.tolist()
    }

def export_prediction(data, result):
    export_data = {
        'timestamp': datetime.now().isoformat(),
        'input_data': data,
        'prediction': {
            'class': result['prediction_label'],
            'confidence': result['probability'],
            'raw_prediction': result['prediction']
        }
    }
    return json.dumps(export_data, indent=2)

def reset_session_state():
    for key in ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']:
        if key in st.session_state:
            del st.session_state[key]

# Load model
model_components = load_model()

# Define options
buying_options = list(buying_map.keys())
maint_options = list(maint_map.keys())
doors_options = list(doors_map.keys())
persons_options = list(persons_map.keys())
lug_boot_options = list(lug_boot_map.keys())
safety_options = list(safety_map.keys())

# App title
st.title("🚗 Car Evaluation Prediction App")
st.markdown("Predict car acceptability class based on categorical features")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Input Car Features")
    with st.form("prediction_form"):
        col_a, col_b = st.columns(2)

        with col_a:
            buying = st.selectbox("Buying Price", buying_options, key="buying")
            doors = st.selectbox("Number of Doors", doors_options, key="doors")
            lug_boot = st.selectbox("Luggage Boot Size", lug_boot_options, key="lug_boot")

        with col_b:
            maint = st.selectbox("Maintenance Cost", maint_options, key="maint")
            persons = st.selectbox("Capacity (Persons)", persons_options, key="persons")
            safety = st.selectbox("Safety Level", safety_options, key="safety")

        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            predict_button = st.form_submit_button("🔮 Predict", type="primary")
        with col_btn2:
            reset_button = st.form_submit_button("🔄 Reset")
        with col_btn3:
            export_button = st.form_submit_button("📤 Export Last Result")

# Reset handler
if reset_button:
    reset_session_state()
    st.rerun()

# Predict handler
if predict_button:
    input_data = {
        'buying': buying,
        'maint': maint,
        'doors': doors,
        'persons': persons,
        'lug_boot': lug_boot,
        'safety': safety
    }

    try:
        result = predict_car(input_data, model_components)
        st.session_state['last_prediction'] = {'input_data': input_data, 'result': result}

        with col2:
            st.subheader("🎯 Prediction Results")
            st.markdown(f"**Predicted Class:** `{result['prediction_label']}`")

            # Gauge
            confidence = result['probability'] * 100
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                title={'text': "Confidence (%)"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "green"}}
            ))
            fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Probability chart
            prob_df = pd.DataFrame({
                'Class': list(class_map.keys()),
                'Probability': result['probabilities']
            })

            fig_bar = px.bar(prob_df, x='Class', y='Probability', color='Probability',
                             color_continuous_scale='viridis',
                             title='Class Probability Distribution')
            fig_bar.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_bar, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Prediction Error: {str(e)}")

# Feature importance
st.subheader("📊 Feature Importance")
if 'model' in model_components:
    try:
        importance_df = pd.DataFrame({
            'Feature': model_components['feature_names'],
            'Importance': model_components['model'].feature_importances_
        }).sort_values('Importance', ascending=True)

        fig_imp = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                         title='Feature Importance', color='Importance',
                         color_continuous_scale='plasma')
        fig_imp.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_imp, use_container_width=True)

    except Exception as e:
        st.error(f"Error displaying feature importance: {str(e)}")

# Export
if export_button:
    if 'last_prediction' in st.session_state:
        export_data = export_prediction(
            st.session_state['last_prediction']['input_data'],
            st.session_state['last_prediction']['result']
        )
        st.download_button(
            label="📥 Download Prediction Results",
            data=export_data,
            file_name=f"car_evaluation_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    else:
        st.warning("⚠️ No prediction results to export. Please make a prediction first.")

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit • Car Evaluation App*")
