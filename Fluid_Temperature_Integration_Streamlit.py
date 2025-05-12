# === Piping Data ===
with col3:
    st.header("Piping Data")
    d = st.number_input("Inner pipe diameter (m):", min_value=0.01, value=0.25716)
    D = st.number_input("Outer pipe diameter (m):", min_value=0.01, value=0.3238)
    L = st.number_input("Pipe length (m):", min_value=1.0, value=40.0)

    st.subheader("Pipe Insulation")
    use_insulation = st.checkbox("Include insulation?", value=False)
    if use_insulation:
        insulation_thickness = st.number_input("Insulation thickness (m):", min_value=0.001, value=0.05)
        k_ins = st.number_input("Insulation thermal conductivity (W/m·K):", min_value=0.001, value=0.035)

# === Run Simulation ===
if st.button("Run Simulation"):
    # Converted Values
    pump_heat_factor = 0.5  # % of lost power to fluid
    dWp_dt = pump_power_kw * (1 - pump_eff / 100) * pump_heat_factor * 1000 * num_pumps  # W
    F = (pump_flow_m3h / 3600) * num_pumps  # m³/s

    m = total_volume_m3 * rho  # kg
    k_pipe = 45  # W/m.K
    h_out = 25  # W/m2.K
    n = 0.33

    # Thermal Resistances
    R_conv_in = 1 / (0.023 * k_fluid * (4 * F * rho / (np.pi * d * mu))**0.8 * (mu * cp_fluid / k_fluid)**n * np.pi * L)
    R_cond_pipe = np.log(D / d) / (2 * np.pi * k_pipe * L)

    # Optional insulation
    if use_insulation:
        D_ins = D + 2 * insulation_thickness
        R_insulation = np.log(D_ins / D) / (2 * np.pi * k_ins * L)
        R_conv_out = 1 / (h_out * np.pi * D_ins * L)
        R_total = R_conv_in + R_cond_pipe + R_insulation + R_conv_out
    else:
        R_conv_out = 1 / (h_out * np.pi * D * L)
        R_total = R_conv_in + R_cond_pipe + R_conv_out

    # Equilibrium temps
    T_eq = T_ambient + dWp_dt * R_total
    T_90 = T_ambient + 0.9 * (T_eq - T_ambient)

    # Euler Simulation
    dt = 0.1
    t_max = t_max_h * 3600
    time = np.arange(0, t_max, dt)
    Tf = np.zeros_like(time)
    Tf[0] = T_ambient

    for i in range(1, len(time)):
        dT_dt = (dWp_dt - (Tf[i-1] - T_ambient) / R_total) / (m * cp_fluid)
        Tf[i] = Tf[i-1] + dT_dt * dt

    # Find 90% time
    idx_90 = np.where(Tf >= T_90)[0]
    if len(idx_90) > 0:
        t_90_h = time[idx_90[0]] / 3600
        T_90_actual = Tf[idx_90[0]]
    else:
        t_90_h = None
        T_90_actual = None

    # Results Display
    st.success(f"Equilibrium Temperature: {T_eq:.1f} °C")
    st.info(f"90% of Equilibrium Temp: {T_90:.1f} °C")
    if t_90_h is not None:
        st.info(f"Time to reach 90% equilibrium: ≈ {t_90_h:.2f} h")

    # Interactive Plot using Plotly
    fig = go.Figure()

    label_text = f"""{num_pumps} Pump(s), {pump_power_kw*num_pumps:.1f} kW, {pump_flow_m3h*num_pumps:.1f} m³/h\nTotal Fluid Volume = {total_volume_m3:.1f} m³"""

    # Add the temperature over time curve
    fig.add_trace(go.Scatter(x=time / 3600, y=Tf, mode='lines', name=label_text))

    # Add equilibrium temperature line
    fig.add_trace(go.Scatter(x=[0, t_max_h], y=[T_eq, T_eq], mode='lines', name=f'Equilibrium Temp: {T_eq:.1f} °C', line=dict(color='red', dash='dash')))

    # Add 90% equilibrium temperature line
    fig.add_trace(go.Scatter(x=[0, t_max_h], y=[T_90, T_90], mode='lines', name=f'90% Equilibrium Temp: {T_90:.1f} °C', line=dict(color='green', dash='dot')))

    # Add time to reach 90% equilibrium line
    if t_90_h is not None:
        fig.add_trace(go.Scatter(x=[t_90_h, t_90_h], y=[T_ambient, T_90_actual], mode='lines', name=f'Time to reach 90% equilibrium ≈ {t_90_h:.2f} h', line=dict(color='green', dash='dot')))

    # Update layout
    fig.update_layout(
        title="Temperature Rise Over Time",
        xaxis_title="Time (h)",
        yaxis_title="Temperature (°C)",
        showlegend=True,
        template="plotly_dark"
    )

    # Display interactive plot in Streamlit
    st.plotly_chart(fig)





