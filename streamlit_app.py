# SunWolf's Sentinel forecast function (honored branch of SUPT)
def sentinel_forecast(proxies, geomag_kp=0, historical_matches=None, domain=None, time_steps=100, freq=1.0, damping=0.1, start_date='2025-12-02', ionex_text=None):
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
