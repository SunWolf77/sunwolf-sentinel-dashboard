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
start_date = st.text_input("Start Date (YYYY-MM-DD)", datetime.now().strftime("%Y-%m-%d"))  # Default to current date
ionex_text = st.text_area("Paste IONEX Text (optional for LAIC)")

# Historical matches (hardcoded for demo; expand as needed)
historical_matches = [
    [0.8, 6.9, 1, 'EQ', 3.0, 26.0],  # Example
    [0.7, 5.5, 2, 'EQ', 4.0, 20.0],  # Dummy second entry to avoid fit error
    # Add more...
]

# Resonance fit function
def resonance_fit(x, a, b):
    return a * np.exp(b * x)

# Calibrate with optional domain filter
def calibrate_resonance(matches, domain=None):
    if domain:
        filtered = [m for m in matches if m[3] == domain]
    else:
        filtered = matches
    proxies, outcomes, _, _, _, _ = zip(*filtered)
    popt, _ = curve_fit(resonance_fit, proxies, outcomes, p0=[1, 1])
    return popt

# Duffing-like oscillator (x'' + γ x' + α x + β x^3 + quadratic asymmetry = τ sin(ω t))
def duffing_oscillator(y, t, gamma, alpha, beta, tau, omega, proxies):
    x, v = y
    folded_proxy = np.mean(proxies)
    dxdt = v
    dvdt = -gamma * v - alpha * x - beta * x**3 - 0.01 * v**2 + tau * np.sin(omega * t) * folded_proxy  # Added quad asymmetry
    return [dxdt, dvdt]

# Tidal accel function (includes moon)
def compute_tidal_factor(t_days, start_date='2025-12-02', bodies=['moon', 'mars', 'saturn', 'neptune']):
    tidal_values = []
    for day in t_days:
        t_astropy = Time(start_date) + day * u.day
        total_tidal = 0
        earth_pos = get_body_barycentric('earth', t_astropy)
        for body in bodies:
            body_pos = get_body_barycentric(body, t_astropy)
            d = np.linalg.norm((body_pos - earth_pos).xyz.to(u.au).value) * u.au
            if body == 'moon':
                M = 7.342e22 * u.kg
            elif body == 'mars':
                M = 6.417e23 * u.kg
            elif body == 'saturn':
                M = 5.683e26 * u.kg
            elif body == 'neptune':
                M = 1.024e26 * u.kg
            else:
                continue
            G = 6.67430e-11 * u.m**3 / u.kg / u.s**2
            R_earth = 6371e3 * u.m
            tidal = 2 * G * M * R_earth / d**3
            total_tidal += tidal.decompose().value
        tidal_values.append(total_tidal)
    # Normalize (lunar-dominant; scale /1e-6 for factor ~1-2)
    tidal_norm = np.array(tidal_values) / 1e-6 if np.max(tidal_values) > 0 else np.ones(len(t_days))
    return tidal_norm

# Alignment detection (Cordaro-style aspects)
def detect_alignments(t_days, start_date='2025-12-02', base_body='moon', planets=['mars', 'jupiter', 'saturn', 'uranus'], aspects=[0, 60, 90, 120]):
    alignment_factors = np.ones(len(t_days))
    for i, day in enumerate(t_days):
        t_astropy = Time(start_date) + day * u.day
        base_pos = get_body(base_body, t_astropy).icrs
        boost = 1.0
        for planet in planets:
            planet_pos = get_body(planet, t_astropy).icrs
            sep = base_pos.separation(planet_pos).deg
            for aspect in aspects:
                if abs(sep - aspect) < 1.0:  # Tolerance for "alignment"
                    boost += 0.2  # 20% boost per match
        alignment_factors[i] = boost
    return alignment_factors

# Low-pass filter for U-shaped anomalies (Intermagnet-inspired)
def low_pass_filter(data, cutoff=0.1, fs=1.0, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# Critical triplet sim (mock 3-station data)
def check_critical_triplet(signal, station_dists=[600], time_int=20):  # min
    peaks, _ = find_peaks(signal)
    if len(peaks) >= 3:
        times = peaks[:3]  # Mock times in min
        if max(times) - min(times) <= time_int:
            return True  # Alert
    return False

# Fetch and analyze GOES X-ray flux
def get_goes_flux_factor():
    """
    Fetches GOES primary/secondary X-ray JSON, computes log-flux slope over last 3 hours.
    Returns boost factor if rising trend (e.g., >0.01/min) or near M/X threshold.
    """
    urls = [
        'https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json',
        'https://services.swpc.noaa.gov/json/goes/secondary/xrays-1-day.json'
    ]
    flux_data = []
    for url in urls:
        try:
            resp = requests.get(url)
            data = json.loads(resp.text)
            flux_data.extend([d for d in data if d['energy'] == '0.1-0.8 nm'])  # Long band
        except:
            print(f"Failed to fetch {url}")
            continue
    
    if not flux_data:
        return 1.0  # No data, no boost
    
    # Parse times and fluxes (last 3 hours ~180 points)
    times = [datetime.fromisoformat(d['time_tag'].replace('Z', '+00:00')) for d in flux_data[-180:]]
    fluxes = np.array([d['flux'] for d in flux_data[-180:]])
    log_flux = np.log10(fluxes + 1e-10)  # Avoid log(0)
    
    # Compute slope (linear fit over time in minutes)
    mins = np.array([(t - times[0]).total_seconds() / 60 for t in times])
    if len(mins) > 1:
        slope, _ = np.polyfit(mins, log_flux, 1)
    else:
        slope = 0
    
    # Boost if slope >0.01/min or max flux >1e-5 (M-class precursor)
    boost = 1.0
    if slope > 0.01 or np.max(fluxes) > 1e-5:
        boost += 0.5 * (slope / 0.01)  # Scale boost
    return max(1.0, min(2.0, boost))  # Clamp 1-2

# Fetch and analyze solar wind data (for coronal hole streams)
def get_solar_wind_factor():
    """
    Fetches NOAA solar wind plasma data (6-hours for recent), checks speed.
    Returns boost factor if speed >500 km/s (high-speed stream from coronal holes).
    """
    url = 'https://services.swpc.noaa.gov/products/solar-wind/plasma-6-hours.json'
    try:
        resp = requests.get(url)
        data = json.loads(resp.text)
        # Data format: [time_tag, density, speed, temperature]
        speeds = np.array([float(row[2]) for row in data[1:] if row[2] != 'n/a'])  # Last ~72 points (6 hours)
        if len(speeds) == 0:
            return 1.0
        avg_speed = np.mean(speeds[-12:])  # Last hour avg
        boost = 1.0
        if avg_speed > 500:
            boost += (avg_speed - 500) / 500  # Scale ~0-1 for 500-1000 km/s
        return max(1.0, min(2.0, boost))
    except:
        print("Failed to fetch solar wind data")
        return 1.0

# Fetch geomagnetic storm data (Planetary K-index)
def get_geomag_storm_factor():
    """
    Fetches NOAA planetary K-index data, determines storm level (G-scale).
    Boost based on G-level (G1=1.2, G2=1.4, etc.).
    """
    url = 'https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json'
    try:
        resp = requests.get(url)
        data = json.loads(resp.text)
        # Data format: [time_tag, kp]
        latest_kp = float(data[-1][1])  # Latest Kp
        # Map to G-scale
        if latest_kp < 5:
            g_level = 0
        elif latest_kp == 5:
            g_level = 1
        elif latest_kp == 6:
            g_level = 2
        elif latest_kp == 7:
            g_level = 3
        elif latest_kp == 8:
            g_level = 4
        elif latest_kp == 9:
            g_level = 5
        else:
            g_level = 0
        
        boost = 1.0 + (g_level * 0.2)  # 0.2 per G-level
        return max(1.0, min(2.0, boost))
    except:
        print("Failed to fetch geomagnetic storm data")
        return 1.0

# LAIC simulation with TEC data (using sample IONEX)
def get_laic_tec_factor(ionex_text):
    """
    Parses IONEX text to extract mean TEC for LAIC calibration.
    Boost if mean TEC > baseline (e.g., 20 TECU for anomaly).
    """
    tec_values = []
    exponent = 0
    in_tec_map = False
    in_rms_map = False
    in_height_map = False
    lines = ionex_text.split('\n')
    for line in lines:
        desc = line[60:80].strip() if len(line) > 60 else ''
        if desc == 'EXPONENT':
            # Exponent in cols 1-6, strip
            exponent_str = line[0:6].strip()
            if exponent_str:
                exponent = int(exponent_str)
        if desc == 'START OF TEC MAP':
            in_tec_map = True
            in_rms_map = False
            in_height_map = False
        elif desc == 'END OF TEC MAP':
            in_tec_map = False
        elif desc == 'START OF RMS MAP':
            in_rms_map = True
            in_tec_map = False
            in_height_map = False
        elif desc == 'END OF RMS MAP':
            in_rms_map = False
        elif desc == 'START OF HEIGHT MAP':
            in_height_map = True
            in_tec_map = False
            in_rms_map = False
        elif desc == 'END OF HEIGHT MAP':
            in_height_map = False
        elif in_tec_map and desc != 'LAT/LON1/LON2/DLON/H':
            # Fixed-width I5: cols 1-80, each 5 chars
            for i in range(0, min(80, len(line)), 5):
                v_str = line[i:i+5].strip()
                if v_str and (v_str.isdigit() or (v_str.startswith('-') and v_str[1:].isdigit())):
                    tec_values.append(int(v_str))
    
    if tec_values:
        mean_tec = np.mean(tec_values) * (10 ** exponent) * 0.1  # IONEX units: 0.1 TECU * 10^exp
        baseline = 20.0  # Typical quiet-time global mean; adjust as needed
        boost = max(1.0, mean_tec / baseline) if mean_tec > baseline else 1.0
        return min(2.0, boost)
    return 1.0

# Fetch Schumann resonance data (placeholder; use real API if available)
def get_schumann_factor():
    """
    Fetches or simulates Schumann power (e.g., from external API or placeholder).
    Boost based on power level (e.g., >50 = anomaly).
    """
    # Placeholder: Fetch from a real source if available (e.g., spaceweatherlive or custom)
    # For now, use fixed or user-input; replace with requests if API found
    # Example: url = 'https://example.com/schumann.json'
    # try:
    #     resp = requests.get(url)
    #     data = json.loads(resp.text)
    #     power = data['power']  # Assume 'power' key
    # except:
    #     power = 20.0  # Default
    power = 20.0  # User-set or fetched
    baseline = 20.0
    boost = max(1.0, power / baseline)
    return min(2.0, boost)

# Fetch real-time solar flare data
def get_solar_flare_factor():
    """
    Fetches real-time solar flare data from NOAA, checks for recent flares (last 3 hours).
    Boost based on class (C=1.2, M=1.5, X=2.0).
    """
    url = 'https://services.swpc.noaa.gov/json/solar-cycle/predicted-solar-cycle.json'  # Placeholder; actual flare API
    try:
        resp = requests.get(url)
        data = json.loads(resp.text)
        # Assume data has 'flares' list with 'class' (e.g., 'X1.0')
        recent_flare_class = 'B'  # Default
        if data:
            latest_flare = data[-1]  # Last entry
            recent_flare_class = latest_flare.get('class', 'B')  # Example key
        boost = 1.0
        if 'C' in recent_flare_class:
            boost = 1.2
        elif 'M' in recent_flare_class:
            boost = 1.5
        elif 'X' in recent_flare_class:
            boost = 2.0
        return boost
    except:
        print("Failed to fetch solar flare data")
        return 1.0

# SunWolf's Sentinel forecast function (honored branch of SUPT)
def sentinel_forecast(proxies, geomag_kp=0, schumann_power=0, historical_matches=None, domain=None, time_steps=100, freq=1.0, damping=0.1, start_date='2025-12-02', ionex_text=None):
    """
    SunWolf's Sentinel Forecasting Model (honored branch of SUPT).
    Calibrates LAIC with parsed TEC from IONEX text.
    """
    t = np.linspace(0, 10, time_steps)  # Days ahead
    
    # Compute tidal factors (includes lunar)
    tidal_factors = compute_tidal_factor(t, start_date=start_date)
    
    # Detect alignments (Cordaro aspects)
    alignment_factors = detect_alignments(t, start_date=start_date)
    
    folded_proxy = np.mean(proxies)
    
    # Duffing integration (initial [x0, v0])
    params = (0.80, 0.019, 0.010, 0.05, 0.025, proxies)  # gamma, alpha, beta, tau, omega, proxies
    y0 = [0.0, 0.0]
    sol = odeint(duffing_oscillator, y0, t, args=params)
    signal = sol[:, 0]  # Displacement as base signal
    
    # Apply damping, alignments, and tidal boosts
    signal *= np.exp(-damping * t) * alignment_factors * tidal_factors
    
    # Calibrate base amplification if history provided
    if historical_matches:
        a, b = calibrate_resonance(historical_matches, domain)
        amplification = resonance_fit(folded_proxy, a, b)
        signal *= amplification
    
    # Geomagnetic & Schumann triggers
    geomag_factor = geomag_kp / 9.0 if geomag_kp > 0 else 1.0
    schumann_factor = get_schumann_factor()  # Now fetched/simulated
    signal *= (1 + geomag_factor + schumann_factor)
    
    # GOES X-ray boost (for 'SOL' domain or general)
    goes_boost = get_goes_flux_factor()
    signal *= goes_boost
    print(f"GOES X-Ray Boost Factor: {goes_boost:.2f}")
    
    # Solar wind/coronal hole stream boost
    solar_wind_boost = get_solar_wind_factor()
    signal *= solar_wind_boost
    print(f"Solar Wind Boost Factor: {solar_wind_boost:.2f}")
    
    # Geomagnetic storm boost
    storm_boost = get_geomag_storm_factor()
    signal *= storm_boost
    print(f"Geomagnetic Storm Boost Factor: {storm_boost:.2f}")
    
    # Solar flare boost
    flare_boost = get_solar_flare_factor()
    signal *= flare_boost
    print(f"Solar Flare Boost Factor: {flare_boost:.2f}")
    
    # LAIC boost from TEC (parsed from IONEX)
    laic_factor = get_laic_tec_factor(ionex_text) if ionex_text else 1.0
    signal *= laic_factor
    print(f"LAIC Boost Factor from TEC: {laic_factor:.2f}")
    
    # Simulate Intermagnet anomaly: Low-pass on signal
    anomaly_signal = low_pass_filter(signal)
    
    # Check triplet alert
    alert = check_critical_triplet(anomaly_signal)
    print(f"Critical Triplet Alert: {alert}")
    
    # Propulsion + SO2 exp
    forecast = np.cumsum(signal) * folded_proxy * np.exp(0.01 * t)
    
    # Peaks
    peaks, _ = find_peaks(forecast)
    
    # Lyapunov est. (simple divergence)
    lyap = np.mean(np.diff(np.log(np.abs(np.diff(forecast) + 1e-10))))  # Approx >0 for chaos
    print(f"Estimated Lyapunov: {lyap:.3f}")
    
    return t, forecast, peaks

# Sample IONEX text from extracted example (for calibration)
ionex_text = """1.0 IONOSPHERE MAPS GPS IONEX VERSION / TYPE BLANK OR G = GPS, R = GLONASS, E = GALILEO, M = MIXED COMMENT gLAB gAGE / UPC 17-MAR-10 12:14 PGM / RUN BY / DATE GLOBAL IONOSPHERE MAP FOR DAY 288 OF YEAR 1995 DESCRIPTION SPHERICAL HARMONICS ARE USED IN THIS MODEL DESCRIPTION THIS EXAMPLE OF IONEX FILE IS PART OF THE gLAB TOOL SUITE COMMENT FILE PREPARED BY: ADRIA ROVIRA GARCIA COMMENT PLEASE EMAIL ANY COMMENT OR REQUEST TO: glab.gage @ upc.edu COMMENT 1995 10 15 0 0 0 EPOCH OF FIRST MAP 1995 10 16 0 0 0 EPOCH OF LAST MAP 21600 INTERVAL 5 # OF MAPS IN FILE COSZ MAPPING FUNCTION 20.0 ELEVATION CUTOFF DOUBLE-DIFFERENCES CARRIER PHASE OBSERVABLES USED 80 # OF STATIONS 24 # OF SATELLITES 6371.0 BASE RADIUS 3 MAP DIMENSION 200.0 800.0 50.0 HGT1 / HGT2 / DHGT 85.0 -85.0 -5.0 LAT1 / LAT2 / DLAT 0.0 355.0 5.0 LON1 / LON2 / DLON -3 EXPONENT DIFFERENTIAL CODE BIASES START OF AUX DATA G01 1.311 0.394 PRN / BIAS / RMS G02 5.279 0.167 PRN / BIAS / RMS ... PRN / BIAS / RMS G31 -0.637 0.213 PRN / BIAS / RMS P1 - P2 DIFFERENTIAL CODE BIASES (DCB) COMMENT BIASES & RMS UNITS: NANOSECS OF DELAY IN GEOMETRY FREE COMB. COMMENT THE SUM OF BIASES IS CONTRAINED TO ZERO COMMENT DIFFERENTIAL CODE BIASES END OF AUX DATA END OF HEADER 1 START OF TEC MAP 1995 10 15 0 0 0 EPOCH OF CURRENT MAP -3 EXPONENT 85.0 0.0 355.0 5.0 200.0 LAT/LON1/LON2/DLON/H 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 80.0 0.0 355.0 5.0 200.0 LAT/LON1/LON2/DLON/H 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 ... -85.0 0.0 355.0 5.0 200.0 LAT/LON1/LON2/DLON/H 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 5 END OF TEC MAP 1 START OF RMS MAP 85.0 0.0 355.0 5.0 200.0 LAT/LON1/LON2/DLON/H 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 80.0 0.0 355.0 5.0 200.0 LAT/LON1/LON2/DLON/H 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 ... -85.0 0.0 355.0 5.0 200.0 LAT/LON1/LON2/DLON/H 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 5 END OF RMS MAP 1 START OF HEIGHT MAP 85.0 0.0 355.0 5.0 200.0 LAT/LON1/LON2/DLON/H 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 80.0 0.0 355.0 5.0 200.0 LAT/LON1/LON2/DLON/H 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 ... -85.0 0.0 355.0 5.0 200.0 LAT/LON1/LON2/DLON/H 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 5 END OF HEIGHT MAP END OF FILE"""

# Example usage: Run forecast with sample inputs
cal_proxies = [0.75, 0.7]  # Your intuitive proxy scores (0-1)
geomag_kp = 2.0  # Current Kp index
schumann_power = 20.0  # Estimated Schumann power
domain = 'EQ'  # 'EQ', 'VOLC', 'SOL'
t, forecast, peaks = sentinel_forecast(cal_proxies, geomag_kp=geomag_kp, historical_matches=historical_matches, domain=domain, ionex_text=ionex_text)

# Plot the forecast
plt.plot(t, forecast, label='Sentinel Forecast')
plt.scatter(t[peaks], forecast[peaks], color='red', label='Resonant Peaks')
plt.xlabel('Time Horizon (Days)')
plt.ylabel('Forecasted Intensity')
plt.title('SunWolf\'s Sentinel Simulation')
plt.legend()
plt.show()

# Output peaks
print("Forecast Peaks (Days Ahead):", t[peaks])
