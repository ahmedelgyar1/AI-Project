import streamlit as st
import pandas as pd
import numpy as np
from real_model import predict_home_automation, load_models, handle_unknown_values, mood_mapping, condition_mapping, time_mapping
import datetime

# Initialize the model
@st.cache_resource
def initialize_model():
    if load_models():
        st.success("Model loaded successfully!")
        return True
    else:
        st.error("Failed to load model!")
        return False

# Set page config
st.set_page_config(
    page_title="Smart Home Automation",
    page_icon="🏠",
    layout="wide"
)

# Title and description
st.title("🏠 Smart Home Automation System")
st.markdown("""
This application predicts optimal home automation settings based on your current state and preferences.
The system will suggest appropriate brightness levels for each room's lighting.
""")

# Initialize model
if not initialize_model():
    st.error("Please check your model files and try again.")
    st.stop()

# Add tabs for different modes
tab1, tab2 = st.tabs(["Standard Mode", "Custom Testing Mode"])

with tab1:
    # Standard Mode (Keep exactly the same as original)
    st.header("Standard Input Parameters")

    col1, col2 = st.columns(2)

    with col1:
        mood = st.selectbox(
            "Mood",
            ["peaceful", "focused", "tired", "stressed", "happy", "calm", "energetic"],
            key="mood1"
        )
        
        person_condition = st.selectbox(
            "Current Condition",
            ["sleeping", "at_work", "at_home", "out", "getting_ready", "awake"],
            key="condition1"
        )
        
        time_of_day = st.selectbox(
            "Time of Day",
            ["morning", "afternoon", "evening", "night"],
            key="time1"
        )

    with col2:
        at_home = st.radio(
            "Are you at home?",
            ["Yes", "No"],
            key="home1"
        )
        
        is_holiday = st.radio(
            "Is it a holiday?",
            ["Yes", "No"],
            key="holiday1"
        )

    # Convert inputs to model format
    at_home_val = 1 if at_home == "Yes" else 0
    is_holiday_val = 1 if is_holiday == "Yes" else 0

    if st.button("Get Recommendations", key="predict1"):
        try:
            recommendations = predict_home_automation(
                mood=mood,
                person_condition=person_condition,
                time_of_day=time_of_day,
                at_home=at_home_val,
                is_holiday=is_holiday_val
            )
            
            if recommendations:
                st.success("Here are your recommended actions:")
                for i, action in enumerate(recommendations, 1):
                    st.write(f"{i}. {action}")
            else:
                st.error("Failed to generate recommendations. Please try again.")
        except ValueError as e:
            st.error(f"Invalid input: {str(e)}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

with tab2:
    st.header("Custom Testing Mode")
    st.markdown("""
    In this mode, you can test the model with custom scenarios. 
    Enter any values and the system will try to understand them.
    """)

    custom_col1, custom_col2 = st.columns(2)

    with custom_col1:
        custom_mood = st.text_input(
            "Custom Mood",
            placeholder="Try: 'zen', 'annoyed', 'sleepy'",
            help="Enter any mood description"
        )

        custom_condition = st.text_input(
            "Custom Condition",
            placeholder="Try: 'gaming', 'studying', 'shopping'",
            help="Enter any activity/condition"
        )

        custom_time = st.text_input(
            "Custom Time of Day",
            placeholder="Try: 'dawn', 'noon', 'midnight'",
            help="Enter any time description"
        )

    with custom_col2:
        custom_at_home = st.radio(
            "At Home Status",
            ["Yes", "No"],
            key="home2"
        )

        custom_is_holiday = st.radio(
            "Holiday Status",
            ["Yes", "No"],
            key="holiday2"
        )

    st.info("""
    💡 The system will automatically map your custom inputs to the closest known values.
    """)

    if st.button("Test Custom Scenario", key="predict2"):
        try:
            # Prepare mapping information for display
            mapped_mood = handle_unknown_values(custom_mood if custom_mood else "happy", 
                                              mood_mapping, 'happy')
            mapped_condition = handle_unknown_values(custom_condition if custom_condition else "at_home", 
                                                    condition_mapping, 'at_home')
            mapped_time = handle_unknown_values(custom_time if custom_time else "afternoon", 
                                              time_mapping, 'afternoon')
            
            # Get recommendations
            recommendations = predict_home_automation(
                mood=custom_mood if custom_mood else "happy",
                person_condition=custom_condition if custom_condition else "at_home",
                time_of_day=custom_time if custom_time else "afternoon",
                at_home=1 if custom_at_home == "Yes" else 0,
                is_holiday=1 if custom_is_holiday == "Yes" else 0
            )
            
            if recommendations:
                st.success("Recommended Actions:")
                for i, action in enumerate(recommendations, 1):
                    st.write(f"{i}. {action}")
                
            else:
                st.error("Failed to generate recommendations.")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Keep original about section
st.markdown("""
---
### About the Model
This smart home automation system uses machine learning to predict optimal settings for:
- Device control (TV, security systems)
- Lighting control with smart brightness
- Music recommendations

The model considers your:
- Current mood and activity
- Time of day and location
- Special days status
""")