import streamlit as st
import pickle
import pandas as pd

# Load the saved model
with open("xgboost_machine_learning_final_model.sav", "rb") as file:
    model = pickle.load(file)

# Feature list hardcoded — no pkl needed
feature_cols = [
    'was_suppressed', 'spike', 'svi_unemp_rate', 'svi_no_hs_diploma',
    'svi_uninsured', 'svi_age65_plus', 'svi_age17_under', 'svi_disabled',
    'svi_single_parent', 'svi_minority', 'svi_limited_english', 'svi_multi_unit',
    'svi_mobile_homes', 'svi_crowded_housing', 'svi_no_vehicle', 'svi_group_quarters',
    'RPL_THEME1', 'RPL_THEME2', 'RPL_THEME3', 'RPL_THEME4', 'RPL_THEMES',
    'svi_poverty_rate', 'svi_housing_burden', 'acs_poverty_rate',
    'acs_unemp_rate', 'acs_median_income'
]

# Function to make predictions
def predict_violence(model, input_df):
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0]
    return ("High Violence Risk" if prediction[0] == 1 else "Low Violence Risk"), probability

# Streamlit app
def main():
    st.title("Early Warning System for County-Level Gun Violence")
    st.write("Enter the socioeconomic details below to predict gun violence risk level:")

    # Input fields
    svi_unemp_rate      = st.number_input("Unemployment Rate (%)",           min_value=0.0, max_value=100.0, value=5.5)
    svi_no_hs_diploma   = st.number_input("No HS Diploma Rate (%)",          min_value=0.0, max_value=100.0, value=12.0)
    svi_uninsured       = st.number_input("Uninsured Rate (%)",               min_value=0.0, max_value=100.0, value=9.0)
    svi_age65_plus      = st.number_input("Age 65+ Rate (%)",                 min_value=0.0, max_value=100.0, value=18.0)
    svi_age17_under     = st.number_input("Age 17 & Under Rate (%)",          min_value=0.0, max_value=100.0, value=22.0)
    svi_disabled        = st.number_input("Disability Rate (%)",              min_value=0.0, max_value=100.0, value=16.0)
    svi_single_parent   = st.number_input("Single Parent Rate (%)",           min_value=0.0, max_value=100.0, value=6.0)
    svi_minority        = st.number_input("Minority Population (%)",          min_value=0.0, max_value=100.0, value=25.0)
    svi_limited_english = st.number_input("Limited English (%)",              min_value=0.0, max_value=100.0, value=1.5)
    svi_multi_unit      = st.number_input("Multi-Unit Housing (%)",           min_value=0.0, max_value=100.0, value=5.0)
    svi_mobile_homes    = st.number_input("Mobile Homes (%)",                 min_value=0.0, max_value=100.0, value=10.0)
    svi_crowded_housing = st.number_input("Crowded Housing (%)",              min_value=0.0, max_value=100.0, value=2.0)
    svi_no_vehicle      = st.number_input("No Vehicle (%)",                   min_value=0.0, max_value=100.0, value=6.0)
    svi_group_quarters  = st.number_input("Group Quarters (%)",               min_value=0.0, max_value=100.0, value=2.0)
    svi_poverty_rate    = st.number_input("Poverty Rate - SVI (%)",           min_value=0.0, max_value=100.0, value=22.0)
    svi_housing_burden  = st.number_input("Housing Burden (%)",               min_value=0.0, max_value=100.0, value=22.0)
    acs_poverty_rate    = st.number_input("Poverty Rate - ACS (%)",           min_value=0.0, max_value=100.0, value=14.0)
    acs_unemp_rate      = st.number_input("Unemployment Rate - ACS (%)",      min_value=0.0, max_value=100.0, value=5.0)
    acs_median_income   = st.number_input("Median Household Income ($)",      min_value=0.0, max_value=250000.0, value=57000.0)
    RPL_THEME1          = st.slider("RPL Theme 1 - Socioeconomic (0-1)",      0.0, 1.0, 0.52)
    RPL_THEME2          = st.slider("RPL Theme 2 - Household (0-1)",          0.0, 1.0, 0.51)
    RPL_THEME3          = st.slider("RPL Theme 3 - Minority/Language (0-1)",  0.0, 1.0, 0.51)
    RPL_THEME4          = st.slider("RPL Theme 4 - Housing/Transport (0-1)",  0.0, 1.0, 0.53)
    RPL_THEMES          = st.slider("RPL Themes - Overall (0-1)",             0.0, 1.0, 0.52)
    was_suppressed      = st.selectbox("Was CDC Count Suppressed?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    spike               = st.selectbox("Spike in Violence Detected?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    # Prepare features for prediction
    input_data = {
        'was_suppressed':      was_suppressed,
        'spike':               spike,
        'svi_unemp_rate':      svi_unemp_rate,
        'svi_no_hs_diploma':   svi_no_hs_diploma,
        'svi_uninsured':       svi_uninsured,
        'svi_age65_plus':      svi_age65_plus,
        'svi_age17_under':     svi_age17_under,
        'svi_disabled':        svi_disabled,
        'svi_single_parent':   svi_single_parent,
        'svi_minority':        svi_minority,
        'svi_limited_english': svi_limited_english,
        'svi_multi_unit':      svi_multi_unit,
        'svi_mobile_homes':    svi_mobile_homes,
        'svi_crowded_housing': svi_crowded_housing,
        'svi_no_vehicle':      svi_no_vehicle,
        'svi_group_quarters':  svi_group_quarters,
        'RPL_THEME1':          RPL_THEME1,
        'RPL_THEME2':          RPL_THEME2,
        'RPL_THEME3':          RPL_THEME3,
        'RPL_THEME4':          RPL_THEME4,
        'RPL_THEMES':          RPL_THEMES,
        'svi_poverty_rate':    svi_poverty_rate,
        'svi_housing_burden':  svi_housing_burden,
        'acs_poverty_rate':    acs_poverty_rate,
        'acs_unemp_rate':      acs_unemp_rate,
        'acs_median_income':   acs_median_income,
    }

    input_df = pd.DataFrame([input_data])[feature_cols]

    # Prediction
    if st.button("Predict"):
        result, probability = predict_violence(model, input_df)
        st.write(f"Prediction: {result}")
        st.write(f"Probability — Low Risk:  {probability[0]*100:.1f}%")
        st.write(f"Probability — High Risk: {probability[1]*100:.1f}%")

if __name__ == "__main__":
    main()
