import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Streamlit App Title
st.title("Pump Heat Simulation Tool")
st.write("Enter your pump and system parameters below:")

# User Inputs
pump_power_kw = st.number_input("Pump power per pump (kW):", min_value=0.1, value=5.0)
pump_flow_m3h = st.number_input("Pump flow rate per pump (m³/h):", min_value=0.1, value=10.0)
pump_eff = st.number_input("Pump efficiency (%):", min_value=1.0, max_value=100.0, value=70.0)
num_pumps = st.number_input("Number of pumps operating in parallel:", min_value=1, step=1, value=1)
t_max_h = st.number_input("Total simulation time (h):", min_value=0.1, value=5.0)

if st.button("Run Simulation"):
    # Converted Values
    pump_heat_factor = 0.5  # % of lost power to fluid
    dWp_dt = pump_power_kw * (1 - pump_eff / 100) * pump_heat_factor * 1000 * num_pumps  # W
    F = (pump_flow_m3h / 3600) * num_pumps  # m³/s

    # Pipe & Fluid Parameters
    total_volume_m3 = 5
    rho = 850  # kg/m3
    m = total_volume_m3 * rho  # kg
    cp_fluid = 2000  # J/kg.K
    T_ambient = 25  # °C
    k_fluid = 0.12  # W/m.K
    mu = 0.3  # Pa.s
    d = 0.25716  # m
    D = 0.3238  # m
    L = 40  # m
    k_pipe = 45  # W/m.K
    h_out = 25  # W/m2.K
    n = 0.33

    # Thermal Resistances
    R_conv_in = 1 / (0.023 * k_fluid * (4 * F * rho / (np.pi * d * mu))**0.8 * (mu * cp_fluid / k_fluid)**n * np.pi * L)
    R_cond_pipe = np.log(D / d) / (2 * np.pi * k_pipe * L)
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
    Tf[0] = 25

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

    # Plotting
    fig, ax = plt.subplots(figsize=(8,5))
    label_text = f"""{num_pumps} Pump(s), {pump_power_kw*num_pumps:.1f} kW, {pump_flow_m3h*num_pumps:.1f} m³/h\nTotal Fluid Volume = {total_volume_m3:.1f} m³"""

    ax.plot(time / 3600, Tf, label=label_text)
    ax.axhline(T_eq, color='red', linestyle='--', label=f'Equilibrium Temp: {T_eq:.1f} °C')
    ax.axhline(T_90, color='green', linestyle=':', label=f'90% Equilibrium Temp: {T_90:.1f} °C')

    if t_90_h is not None:
        ax.axvline(t_90_h, color='green', linestyle=':', label=f'Time to reach 90% equilibrium ≈ {t_90_h:.2f} h')

    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Temperature Rise Over Time')
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)



