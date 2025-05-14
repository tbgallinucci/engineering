import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Streamlit App Title
st.title("Pump Heat Simulation Tool")
st.write("Enter your pump, fluid, piping, and insulation parameters below:")

# === System Data ===
st.header("System Data")
total_volume_m3 = st.number_input("Total fluid volume in system (m³):", min_value=0.1, value=10.0)
T_ambient = st.number_input("Ambient temperature (°C):", value=25.0)
target_mu = st.number_input("Target Viscosity (cP):", value=25.0)
max_mu = target_mu*1.1/1000
min_mu = target_mu*0.9/1000

# === Fluid Data ===
st.header("Fluid Data")
use_manual_input = st.checkbox("Manually input fluid properties")

if use_manual_input:
    rho = st.number_input("Fluid density (kg/m³):", min_value=100.0, value=850.0)
    cp_fluid = st.number_input("Fluid specific heat capacity (J/kg·K):", min_value=0.1, value=2000.0) 
    k_fluid = st.number_input("Fluid thermal conductivity (W/m·K):", min_value=0.01, value=0.12)
    mu_constant = st.number_input("Fluid dynamic viscosity (Pa·s):", min_value=0.001, value=0.3)
    viscosity_model = lambda Tf: mu_constant
else:
    fluid_choice = st.selectbox("Select fluid from library:", [
        "KRD MAX 225",
        "KRD MAX 2205",
        "KRD MAX 685",
        "KRD MAX 55"
    ])
    
    # Fluid properties
    rho = 850.0  # kg/m³
    cp_fluid = 2000.0  # J/kg·K
    k_fluid = 0.12  # W/m·K

    # viscosity models by fluid
    if fluid_choice == "KRD MAX 225":
        viscosity_model = lambda Tf: 0.1651 * np.exp(-0.046 * Tf)

    elif fluid_choice == "KRD MAX 2205":
        viscosity_model = lambda Tf: 1.9133 * np.exp(-0.053 * Tf)

    elif fluid_choice == "KRD MAX 685":
        viscosity_model = lambda Tf: 0.5933 * np.exp(-0.054 * Tf)

    elif fluid_choice == "KRD MAX 55":
        viscosity_model = lambda Tf: -9e-08 * Tf**3 + 1e-05 * Tf**2 - 0.0007 * Tf + 0.0165

# === Pump Data ===

st.header("Heating Phase Pump Config")
pump_power_kw = st.number_input("Nominal power heating per pump (kW):", min_value=0.1, value=69.0)
pump_flow_m3h = st.number_input("Flow rate per heating pump (m³/h):", min_value=0.1, value=550.0)
pump_eff = st.number_input("Heating pump efficiency (%):", min_value=1.0, max_value=100.0, value=58.0)
num_pumps = st.number_input("Number of heating pumps operating in parallel:", min_value=1, step=1, value=1)

st.header("Calibration Phase Pump Config")
calib_pump_power_kw = st.number_input("Nominal power per calibration pump (kW):", min_value=0.1, value=69.0)
calib_pump_flow_m3h = st.number_input("Flow rate per calibration pump (m³/h):", min_value=0.1, value=550.0)
calib_pump_eff = st.number_input("Calibration pump efficiency (%):", min_value=1.0, max_value=100.0, value=58.0)
calib_num_pumps = st.number_input("Number of calibration pumps operating in parallel:", min_value=1, step=1, value=1)

# === Piping Data ===

st.header("Piping Data")
d = st.number_input("Inner pipe diameter (m):", min_value=0.01, value=0.25716)
D = st.number_input("Outer pipe diameter (m):", min_value=0.01, value=0.3238)
L = st.number_input("Pipe length (m):", min_value=1.0, value=40.0)

# Insulation option
use_insulation = st.checkbox("Use pipe insulation?", value=False)

if use_insulation:
    insulation_thickness = st.number_input("Insulation thickness (m):", min_value=0.001, value=0.01)  # e.g., 10mm
    D_insul = D + 2 * insulation_thickness
    st.write(f"Outer diameter with insulation: {D_insul:.3f} m")
    k_insul = st.number_input("Insulation thermal conductivity (W/m·K):", min_value=0.01, value=0.04)


t_max_h = st.number_input("Total simulation time (h):", min_value=0.1, value=24.0)

# === Run Simulation ===
if st.button("Run Simulation"):
    # Convert inputs for Heating Phase
    pump_heat_factor = 1.0  # lost power to fluid
    dWp_dt = pump_power_kw * pump_eff/100 * pump_heat_factor * 1000 * num_pumps  # W
    F = (pump_flow_m3h / 3600) * num_pumps  # m³/s

    m = total_volume_m3 * rho  # kg
    k_pipe = 45  # W/m.K
    h_out = 25  # W/m2.K
    n = 0.33

    # Thermal Resistances
    mu = viscosity_model(T_ambient)
    Re = (4 * F * rho) / (np.pi * d * mu)
    R_conv_in = 1 / (0.023 * k_fluid * Re**0.8 * (mu * cp_fluid / k_fluid)**n * np.pi * L)
    R_cond_pipe = np.log(D / d) / (2 * np.pi * k_pipe * L)
    R_cond_insul = np.log(D_insul / D) / (2 * np.pi * k_insul * L) if use_insulation else 0
    outer_diameter = D_insul if use_insulation else D
    R_conv_out = 1 / (h_out * np.pi * outer_diameter * L)

    # Euler Simulation
    dt = 0.1
    t_max = t_max_h * 3600
    time = np.arange(0, t_max, dt)
    Tf = np.zeros_like(time)
    Tf[0] = T_ambient

    for i in range(1, len(time)):
        mu_t = viscosity_model(Tf[i-1])
        Re = (4 * F * rho) / (np.pi * d * mu_t)
        Pr = (mu_t * cp_fluid) / k_fluid
        Nu = 0.023 * Re**0.8 * Pr**n
        h_in = Nu * k_fluid / d
        R_conv_in = 1 / (h_in * np.pi * d * L)

        R_total = R_conv_in + R_cond_pipe + R_cond_insul + R_conv_out
        dT_dt = (dWp_dt - (Tf[i-1] - T_ambient) / R_total) / (m * cp_fluid)
        Tf[i] = Tf[i-1] + dT_dt * dt

    T_eq = T_ambient + dWp_dt * R_total
    T_90 =  -21.7391 * np.log(min_mu / 0.1651)
    T_110 = -21.7391 * np.log(max_mu / 0.1651)
    T_target = -21.7391 * np.log(target_mu/1000 / 0.1651)

    # Find 110% time
    idx_110 = np.where(Tf >= T_110)[0]
    if len(idx_110) > 0:
        t_110_h = time[idx_110[0]] / 3600  # Convert seconds to hours
        T_110_actual = Tf[idx_110[0]]
    else:
        t_110_h = None
        T_110_actual = None

# Calibration Phase Simulation
# Use conditions at t_110_h and T_110 for calibration phase
    if t_110_h is not None:
        # Truncate the heating phase data at t_110_h
        idx_110_heating = np.where(time <= t_110_h * 3600)[0]  # Find all indices where time <= t_110_h
        time_heating_truncated = time[idx_110_heating]  # Truncated time array for heating phase
        Tf_heating_truncated = Tf[idx_110_heating]  # Truncated temperature array for heating phase

        # Create the adjusted time array for the calibration phase starting from t_110_h
        time_calib = np.arange(t_110_h * 3600, t_max, dt)  # Start from t_110_h in seconds
        Tf_calib = np.zeros_like(time_calib)
        Tf_calib[0] = T_110_actual  # Set the initial temperature for the calibration phase

        # Use the calibration pump configuration for the simulation
        dWp_dt_calib = calib_pump_power_kw * calib_pump_eff/100 * pump_heat_factor * 1000 * calib_num_pumps  # W
        F_calib = (calib_pump_flow_m3h / 3600) * calib_num_pumps  # m³/s

    # Run the simulation for the calibration phase
    for i in range(1, len(time_calib)):
        mu_t_calib = viscosity_model(Tf_calib[i-1])
        Re_calib = (4 * F_calib * rho) / (np.pi * d * mu_t_calib)
        Pr_calib = (mu_t_calib * cp_fluid) / k_fluid
        Nu_calib = 0.023 * Re_calib**0.8 * Pr_calib**n
        h_in_calib = Nu_calib * k_fluid / d
        R_conv_in_calib = 1 / (h_in_calib * np.pi * d * L)

        R_total_calib = R_conv_in_calib + R_cond_pipe + R_cond_insul + R_conv_out
        dT_dt_calib = (dWp_dt_calib - (Tf_calib[i-1] - T_ambient) / R_total_calib) / (m * cp_fluid)
        Tf_calib[i] = Tf_calib[i-1] + dT_dt_calib * dt

    # Display Results for Calibration Phase
    st.write(f"Calibration Phase started at {t_110_h:.2f} hours with temperature {T_110_actual:.2f}°C")

    # Create plot of Temperature over time
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_heating_truncated/3600, y=Tf_heating_truncated, mode='lines', name='Heating Phase'))
    fig.add_trace(go.Scatter(x=time_calib/3600, y=Tf_calib, mode='lines', name='Calibration Phase'))
    fig.update_layout(title="Temperature vs Time", xaxis_title="Time (hours)", yaxis_title="Temperature (°C)")
    st.plotly_chart(fig)

else:
    st.write("Heating Phase did not reach the target temperature within the specified time.")





