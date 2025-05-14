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

    st.header("Pump Data")
    pump_power_kw = st.number_input("Nominal power per pump (kW):", min_value=0.1, value=69.0)
    pump_flow_m3h = st.number_input("Flow rate per pump (m³/h):", min_value=0.1, value=550.0)
    pump_eff = st.number_input("Pump efficiency (%):", min_value=1.0, max_value=100.0, value=58.0)
    num_pumps = st.number_input("Number of pumps operating in parallel:", min_value=1, step=1, value=1)


# === Piping Data ===

    st.header("Piping Data")
    d = st.number_input("Inner pipe diameter (m):", min_value=0.01, value=0.25716)
    D = st.number_input("Outer pipe diameter (m):", min_value=0.01, value=0.3238)
    L = st.number_input("Pipe length (m):", min_value=1.0, value=40.0)

    # Insulation option should not be indented too far
    use_insulation = st.checkbox("Use pipe insulation?", value=False)

    if use_insulation:
        insulation_thickness = st.number_input("Insulation thickness (m):", min_value=0.001, value=0.01)  # e.g., 10mm
        D_insul = D + 2 * insulation_thickness
        st.write(f"Outer diameter with insulation: {D_insul:.3f} m")
        k_insul = st.number_input("Insulation thermal conductivity (W/m·K):", min_value=0.01, value=0.04)


t_max_h = st.number_input("Total simulation time (h):", min_value=0.1, value=24.0)

# === Run Simulation ===
if st.button("Run Simulation"):
    # Converted Values
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

        # Results Display
    mu_eq = viscosity_model(T_eq)
    mu_90 = viscosity_model(T_90)
    mu_110 = viscosity_model(T_110)

    # Find 90% time
    idx_90 = np.where(Tf >= T_90)[0]
    if len(idx_90) > 0:
        t_90_h = time[idx_90[0]] / 3600
        T_90_actual = Tf[idx_90[0]]
    else:
        t_90_h = None
        T_90_actual = None

    # Find 110% time
    idx_110 = np.where(Tf >= T_110)[0]
    if len(idx_110) > 0:
        t_110_h = time[idx_110[0]] / 3600
        T_110_actual = Tf[idx_110[0]]
    else:
        t_110_h = None
        T_110_actual = None

    st.write(f"🛢️ **Selected Fluid**: {fluid_choice}")
    st.write(f"🎯 **Target Viscosity**: {target_mu:.2f} cP")
    st.write(f"🔼 **Temperature for Max Viscosity ({max_mu*1000:.2f} cP)**: {T_110:.1f} °C")
    st.write(f"🔽 **Temperature for Min Viscosity ({min_mu*1000:.2f} cP)**: {T_90:.1f} °C")
    # Convert t_110_h to hours and minutes
    t_110_hours = int(t_110_h)
    t_110_minutes = int((t_110_h - t_110_hours) * 60)

    st.write(f"⏱️ **Time to Max Viscosity**: {t_110_hours} h {t_110_minutes} min")
    # Calculate the time difference in hours
    calibration_time_h = t_90_h - t_110_h

    # Convert to full hours and minutes
    hours = int(calibration_time_h)
    minutes = int((calibration_time_h - hours) * 60)

    st.write(f"📏 **Available Calibration Time after reaching Max Viscosity**: {hours} h {minutes} min")

    # Interactive Plot using Plotly
    fig = go.Figure()

    label_text = f"""{fluid_choice} - {num_pumps} Pump(s), {pump_power_kw*num_pumps:.1f} kW, {pump_flow_m3h*num_pumps:.1f} m³/h\nTotal Fluid Volume = {total_volume_m3:.1f} m³"""

    # Add the temperature over time curve
    fig.add_trace(go.Scatter(x=time / 3600, y=Tf, mode='lines', name=label_text))

    # Add equilibrium temperature line
    fig.add_trace(go.Scatter(x=[0, t_max_h], y=[T_eq, T_eq], mode='lines', 
                             name=f'Equilibrium Temp: {T_eq:.1f} °C', 
                             line=dict(color='red', dash='dash')))

    # Add 90% viscosity temperature line (horizontal)
    fig.add_trace(go.Scatter(x=[0, t_max_h], y=[T_90, T_90], mode='lines', 
                             name=f'90% Viscosity Temp: {T_90:.1f} °C', 
                             line=dict(color='green', dash='dot')))

    # Add time to reach 90% viscosity (vertical, crossing whole plot)
    fig.add_trace(go.Scatter(x=[t_90_h, t_90_h], y=[T_ambient - 5, T_eq + 5], mode='lines', 
                             name=f'Time to reach 90% Viscosity ≈ {t_90_h:.2f} h', 
                             line=dict(color='green', dash='dot')))

    # Add green dot at 90% viscosity
    fig.add_trace(go.Scatter(x=[t_90_h], y=[T_90_actual], mode='markers', 
                             marker=dict(color='green', size=7), 
                             name='90% Viscosity Point'))

    # Add 110% viscosity temperature line (horizontal)
    fig.add_trace(go.Scatter(x=[0, t_max_h], y=[T_110, T_110], mode='lines', 
                             name=f'110% Viscosity Temp: {T_110:.1f} °C', 
                             line=dict(color='purple', dash='dot')))

    # Add time to reach 110% viscosity (vertical, crossing whole plot)
    fig.add_trace(go.Scatter(x=[t_110_h, t_110_h], y=[T_ambient - 5, T_eq + 5], mode='lines', 
                             name=f'Time to reach 110% Viscosity ≈ {t_110_h:.2f} h', 
                             line=dict(color='purple', dash='dot')))

    # Add green dot at 110% viscosity
    fig.add_trace(go.Scatter(x=[t_110_h], y=[T_110_actual], mode='markers', 
                             marker=dict(color='purple', size=7), 
                             name='110% Viscosity Point'))
    
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






