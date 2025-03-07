import streamlit as st

def converter(value, unit_to, unit_from):

    conversion_units = {
          'meters': 1,
        'kilometers': 0.001,
        'centimeters': 100,
        'millimeters': 1000,
        'inches': 39.3701,
        'feet': 3.28084,
        'yards': 1.09361,
        'miles': 0.000621371
    }

    key = f"{unit_from}_{unit_to}" # Generate Single key for conversion units 
    return value * (conversion_units[unit_to] / conversion_units[unit_from])

    
# Streamlit UI

st.title("Convert Your Measuring Units")

value = st.number_input("Enter The Value For Conversion")
unit_from = st.selectbox("Select The Unit to Measure from",['meters','kilometers','centimeters','millimeters', 'inches','feet','yards','miles'])     
unit_to = st.selectbox("Select The Unit to Measure to",['meters','kilometers','centimeters','millimeters', 'inches','feet','yards','miles'])     

if st.button("Convert"):
    result = converter(value, unit_to, unit_from)
    st.write(f"Conversion Value is {result}")