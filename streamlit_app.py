import streamlit as st
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
from scipy.optimize import curve_fit
from scipy.integrate import odeint
from astropy.coordinates import get_body, SkyCoord, get_body_barycentric
from astropy.time import Time
import astropy.units as u
import matplotlib.pyplot as plt
import requests
import json
from datetime import datetime, timedelta

st.title("SunWolf's Sentinel Forecasting Dashboard")

# Inputs
col1, col2 = st.columns(2)
proxy1 = col1.slider("Proxy 1 (0-1)", 0.0, 1.0, 0.75)
proxy2 = col2.slider("Proxy 2 (0-1)", 0.0, 1.0, 0.7)
proxies = [proxy1, proxy2]
geomag_kp = st.number_input("Geomag Kp Index", value=2.0)
domain = st.selectbox("Domain", ['EQ', 'VOLC', 'SOL'])
start_date = st.text_input("Start Date (YYYY-MM-DD)", "2025-12-03")
ionex_text = st.text_area("Paste IONEX Text (optional for LAIC)")

# Historical matches (hardcoded for demo; expand as needed)
historical_matches = [
    [0.8, 6.9, 1, 'EQ', 3.0, 26.0],  # Example
    # Add more...
]

# Model functions (insert full definitions here from our previous exchanges)
# For brevity, assume they're defined: resonance_fit, calibrate_resonance, duffing_oscillator, compute_tidal_factor, detect_alignments, low_pass_filter, check_critical_triplet, get_goes_flux_factor, get_solar_wind_factor, get_geomag_storm_factor, get_laic_tec_factor, get_schumann_factor, get_solar_flare_factor, sentinel_forecast

# Manual refresh buttons for APIs
col_refresh1, col_refresh2, col_refresh3, col_refresh4, col_refresh5, col_refresh6 = st.columns(6)
with col_refresh1:
    if st.button("Refresh GOES X-Ray"):
        try:
            st.session_state.goes_boost = get_goes_flux_factor()
        except Exception as e:
            st.error(f"GOES X-Ray refresh failed: {e}")
            st.session_state.goes_boost = 1.0
with col_refresh2:
    if st.button("Refresh Solar Wind"):
        try:
            st.session_state.solar_wind_boost = get_solar_wind_factor()
        except Exception as e:
            st.error(f"Solar Wind refresh failed: {e}")
            st.session_state.solar_wind_boost = 1.0
with col_refresh3:
    if st.button("Refresh Geomagnetic Storm"):
        try:
            st.session_state.storm_boost = get_geomag_storm_factor()
        except Exception as e:
            st.error(f"Geomagnetic Storm refresh failed: {e}")
            st.session_state.storm_boost = 1.0
with col_refresh4:
    if st.button("Refresh LAIC TEC"):
        try:
            st.session_state.laic_factor = get_laic_tec_factor(ionex_text)  # Refreshes based on input text
        except Exception as e:
            st.error(f"LAIC TEC refresh failed: {e}")
            st.session_state.laic_factor = 1.0
with col_refresh5:
    if st.button("Refresh Schumann"):
        try:
            st.session_state.schumann_factor = get_schumann_factor()  # Refreshes Schumann
        except Exception as e:
            st.error(f"Schumann refresh failed: {e}")
            st.session_state.schumann_factor = 1.0
with col_refresh6:
    if st.button("Refresh Solar Flare"):
        try:
            st.session_state.flare_boost = get_solar_flare_factor()  # Refreshes Solar Flare
        except Exception as e:
            st.error(f"Solar Flare refresh failed: {e}")
            st.session_state.flare_boost = 1.0

# Display current boost values (with defaults if not refreshed)
goes_boost = st.session_state.get('goes_boost', 1.0)
solar_wind_boost = st.session_state.get('solar_wind_boost', 1.0)
storm_boost = st.session_state.get('storm_boost', 1.0)
laic_factor = st.session_state.get('laic_factor', 1.0)
schumann_factor = st.session_state.get('schumann_factor', 1.0)
flare_boost = st.session_state.get('flare_boost', 1.0)

st.write(f"Current GOES X-Ray Boost: {goes_boost:.2f}")
st.write(f"Current Solar Wind Boost: {solar_wind_boost:.2f}")
st.write(f"Current Geomagnetic Storm Boost: {storm_boost:.2f}")
st.write(f"Current LAIC TEC Boost: {laic_factor:.2f}")
st.write(f"Current Schumann Boost: {schumann_factor:.2f}")
st.write(f"Current Solar Flare Boost: {flare_boost:.2f}")

if st.button("Run Forecast"):
    try:
        t, forecast, peaks = sentinel_forecast(proxies, geomag_kp=geomag_kp, historical_matches=historical_matches, domain=domain, start_date=start_date, ionex_text=ionex_text)
        
        fig, ax = plt.subplots()
        ax.plot(t, forecast, label='Forecast')
        ax.scatter(t[peaks], forecast[peaks], color='red', label='Peaks')
        ax.set_xlabel('Days Ahead')
        ax.set_ylabel('Intensity')
        ax.set_title('Sentinel Forecast')
        ax.legend()
        st.pyplot(fig)
        
        st.write("Forecast Peaks (Days Ahead):", t[peaks])
    except Exception as e:
        st.error(f"Forecast run failed: {e}")
