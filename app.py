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

# Model functions (paste all from previous code)
# ... (Omit for brevity; include full definitions for resonance_fit, calibrate_resonance, duffing_oscillator, compute_tidal_factor, detect_alignments, low_pass_filter, check_critical_triplet, get_goes_flux_factor, get_solar_wind_factor, get_geomag_storm_factor, get_laic_tec_factor, get_schumann_factor, get_solar_flare_factor)

if st.button("Run Forecast"):
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
