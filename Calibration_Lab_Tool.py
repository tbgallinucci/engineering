"""
=============================================================================
 CALIBRATION LAB SIZING TOOL
 Engenharia de Processos | Python + Streamlit + Plotly
=============================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import io
import math
from datetime import datetime

st.set_page_config(
    page_title="Calibration Lab Sizing Tool",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# TRANSLATIONS
# ─────────────────────────────────────────────────────────────────────────────
TR = {
    "pt": {
        "app_title": "🏭 Calibration Lab Sizing Tool",
        "app_caption": "Ferramenta Paramétrica de Dimensionamento para Laboratório de Calibração de Medidores de Óleo em Loop Fechado",
        "sidebar_header": "⚙️ Parâmetros Globais do Sistema",
        "fluid_sub": "Fluido",
        "manual_cb": "Inserir propriedades manualmente",
        "density": "Densidade (kg/m³):",
        "cp": "Calor específico (J/kg·K):",
        "k_lbl": "Condutividade térmica (W/m·K):",
        "mu_lbl": "Viscosidade dinâmica (Pa·s):",
        "sel_fluid": "Selecione o fluido:",
        "pipe_sub": "Tubulação (Térmica)",
        "d_in": "Diâmetro interno (m):",
        "d_out": "Diâmetro externo (m):",
        "L_p": "Comprimento total de tubo (m):",
        "eps_lbl": "Emissividade da tubulação (ε):",
        "eps_help": "Aço carbono oxidado ≈ 0.7–0.9 | Aço polido ≈ 0.05–0.1",
        "hout_lbl": "Coef. convecção externa (W/m²·K):",
        "hout_help": "Convecção natural em ar ≈ 5–15 W/m²·K",
        "tab_th": "🌡️ Simulação Térmica",
        "tab_hy": "💧 Curva do Sistema",
        "tab_bm": "📋 Lista de Materiais (BOM)",
        "tab_mn": "📖 Manual / Memorial",
        "th_header": "🌡️ Simulação Térmica – Aquecimento e Calibração",
        "sys_data": "Dados do Sistema",
        "vol": "Volume total de fluido no sistema (m³):",
        "tamb": "Temperatura ambiente (°C):",
        "mu_nom": "Viscosidade nominal de processo (cP):",
        "tsim": "Tempo total de simulação (h):",
        "heat_ph": "⬇️ Fase de Aquecimento",
        "calib_ph": "⬇️ Fase de Calibração",
        "p_kw": "Potência nominal da bomba (kW):",
        "p_kw_help": "Potência mantida constante — variação ao longo da curva da bomba centrífuga é pequena e fica a critério do engenheiro (usar máxima ou nominal).",
        "q_ph": "Vazão total da fase (m³/h):",
        "eff": "Eficiência da bomba (%):",
        "hf": "Fator de calor da bomba:",
        "hf_help": "Fração da potência mecânica convertida em calor no fluido. Loop fechado com bomba centrífuga ≈ 1,0.",
        "run": "▶️ Executar Simulação Térmica",
        "err_mu": "Não foi possível resolver as temperaturas-alvo de viscosidade. Verifique o modelo de fluido e a viscosidade alvo.",
        "warn_110": "A temperatura de início de calibração (110% µ) não foi atingida. Aumente o tempo de simulação ou a potência da bomba.",
        "res_hdr": "Resultados",
        "r_mu": "Viscosidade Alvo",
        "r_Tt": "Temp. Alvo (100% µ)",
        "r_T90": "Temp. 90% µ",
        "r_T110": "Temp. 110% µ",
        "r_Teq": "Temp. Equilíbrio",
        "r_th": "⏱ Tempo de Aquecimento",
        "r_cw": "📏 Janela de Calibração",
        "na": "Não atingida",
        "pl_title": "Temperatura do Fluido vs Tempo",
        "pl_x": "Tempo (h)",
        "pl_y": "Temperatura (°C)",
        "tr_heat": "Fase Aquecimento",
        "tr_calib": "Fase Calibração",
        "cw_lbl": "Janela de Calibração",
        "hy_header": "💧 Curva do Sistema – Hidráulica",
        "hy_pipe": "Parâmetros da Tubulação",
        "hy_d": "Diâmetro interno (m):",
        "hy_rug": "Rugosidade absoluta (mm):",
        "hy_rug_h": "Aço carbono novo ≈ 0.046 mm",
        "hy_L": "Comprimento total de tubo (m):",
        "hy_dz": "Desnível / elevação estática (m):",
        "hy_c90": "Qtd. curvas 90°:",
        "hy_c45": "Qtd. curvas 45°:",
        "hy_tee": "Qtd. tês:",
        "hy_ve": "Qtd. válvulas esfera:",
        "hy_vb": "Qtd. válvulas borboleta de bloqueio (K=0,5):",
        "hy_vb_help": "Válvulas borboleta normalmente abertas usadas para isolamento de ramais (K fixo = 0,5). NÃO inclua aqui as válvulas de controle — elas são configuradas abaixo com Kv variável.",
        "fittings_help": "**Conexões e válvulas de bloqueio** *(perdas localizadas — método K)*\n\n| Componente | K |\n|---|---|\n| Curva 90° | 0,30 |\n| Curva 45° | 0,20 |\n| Tê | 0,50 |\n| Válvula esfera | 0,10 |\n| V. borboleta bloqueio | 0,50 |",
        "hy_global_note": "ℹ️ Diâmetro, comprimento, rugosidade e desnível são definidos nas **Configurações Globais** (barra lateral). Altere lá para refletir aqui.",
        "hy_fittings": "Conexões e Válvulas de Bloqueio",
        "kv_mode_lbl": "Modo da válvula de controle:",
        "kv_mode_single": "Kv único / linear",
        "kv_mode_curve": "Curva Kv × abertura (3 pontos)",
        "kv_curve_help": "Insira 3 pontos da curva do fabricante: abertura (%) e Kv correspondente. A interpolação é log-linear, adequada para válvulas de equal-percentage.",
        "kv_op_j": "Abertura (%)",
        "kv_kv_j": "Kv (m³/h·bar⁰·⁵)",
        "n_ctrl_lbl": "Quantidade de válvulas de controle:",
        "n_ctrl_help": "Cada válvula de controle possui Kv e abertura independentes. Em loop fechado, válvulas em série somam as perdas de pressão.",
        "ctrl_series_note": "ℹ️ As válvulas de controle estão em **paralelo** no loop — os valores de Kv de cada linha são somados.",
        "ctrl_v": "Válvula de Controle",
        "kv_lbl": "Kv da válvula de controle (m³/h·bar⁰·⁵) — IEC 60534:",
        "kv_help": "0 = sem válvula de controle.",
        "op_lbl": "Abertura da válvula (%)",
        "op_help": "100% = totalmente aberta.",
        "fl_hy": "Fluido (para hidráulica)",
        "hy_rho": "Densidade (kg/m³):",
        "hy_mu_h": "Viscosidade nominal de processo (cP):",
        "hy_mu_h_h": "Viscosidade na temperatura nominal de calibração (mesma da aba Simulação Térmica).",
        "hy_qmax": "Vazão máxima da curva (m³/h):",
        "hy_qmax_h": "Limite direito do eixo X do gráfico. Use a vazão máxima do laboratório.",
        "pump_curve_hdr": "Curva da Bomba (5 pontos — rotação nominal)",
        "pump_curve_help": "Insira 5 pontos Q×H da curva do fabricante na rotação nominal. O software aplica as leis de semelhança para calcular as curvas em outras frequências.",
        "pc_poles": "Número de polos do motor:",
        "pc_poles_help": "2 polos → 3600 RPM (60 Hz) | 4 polos → 1800 RPM (60 Hz) | 6 polos → 1200 RPM (60 Hz)",
        "pc_freq": "Frequência nominal da curva fornecida (Hz):",
        "pc_freq_help": "Frequência em que o fabricante mediu os 5 pontos. Geralmente 60 Hz.",
        "pc_fmin": "Frequência mínima do inversor (Hz):",
        "pc_fmax": "Frequência máxima do inversor (Hz):",
        "pc_Q_lbl": "Q{} (m³/h):",
        "pc_H_lbl": "H{} (m):",
        "pc_pt": "Ponto",
        "op_pt_sys": "Pontos de Operação — Rotação Variável",
        "op_Q": "Vazão no ponto de operação",
        "op_H": "Altura manométrica no ponto de operação",
        "sys20": "Sistema — abertura 20%",
        "sys100": "Sistema — abertura 100%",
        "sys_user": "Sistema — abertura definida pelo usuário",
        "pump_nom": "Bomba — {f} Hz (nominal)",
        "pump_fmin": "Bomba — {f} Hz (mín. inversor)",
        "pump_fmax": "Bomba — {f} Hz (máx. inversor)",
        "calc_btn": "📊 Calcular Curva do Sistema",
        "kv_calc": "**Calculadora rápida de Kv:**",
        "kv_Q": "Q (m³/h):",
        "kv_dP": "ΔP máx (bar):",
        "kv_rho": "ρ (kg/m³):",
        "kv_res": "**Kv mínimo: {kv:.0f} m³/h·bar⁰·⁵**",
        "m_H800h": "H total @ 800 m³/h (quente)",
        "m_H800c": "H total @ 800 m³/h (frio)",
        "m_V800": "Velocidade @ 800 m³/h",
        "cold_warn": "⚠️ **Partida a Frio:** Re = {re:.0f} a 800 m³/h com µ fria ({mu:.0f} cP) — escoamento **{reg}**. Verifique potência do motor e torque de partida do inversor.",
        "lam": "laminar", "trans": "transição",
        "tr_hot": "H Total – Quente (nominal)",
        "tr_cold": "H Total – Frio (partida)",
        "tr_dist": "Perda Distribuída – Quente",
        "tr_loc": "Perda Localizada (K)",
        "tr_cv": "Perda Válvula Controle (Kv)",
        "hy_title": "Curva do Sistema – Altura Manométrica vs Vazão",
        "hy_px": "Vazão (m³/h)", "hy_py": "Altura Manométrica (m)",
        "hy_tbl": "Tabela de Perdas em Pontos-Chave",
        "cQ":"Vazão (m³/h)","cHd":"H Distribuída (m)","cHl":"H Localizada (m)",
        "cHe":"H Estática (m)","cHc":"H Controle (m)","cHt":"H TOTAL (m)","cdP":"ΔP (bar)",
        "bm_hdr": "📋 Lista de Materiais (BOM)",
        "bm_info": "Preencha os parâmetros para gerar a BOM.",
        "bm_tv": "Volume do tanque (m³):",
        "bm_DN": "Diâmetro nominal da tubulação (mm):",
        "bm_L": "Comprimento total de tubulação (m):",
        "bm_c90":"Qtd. curvas 90°:","bm_c45":"Qtd. curvas 45°:","bm_tee":"Qtd. tês:",
        "bm_ve":"Qtd. válvulas esfera:","bm_vb":"Qtd. válvulas borboleta:",
        "bm_vc":"Qtd. válvulas de controle:","bm_pt":"Qtd. transmissores de pressão (PT):",
        "bm_tt":"Qtd. transmissores de temperatura (TT):","bm_ftm":"Qtd. Master Meters:",
        "bm_ftc":"Qtd. medidores em calibração:",
        "bm_pump": "Dados da Bomba (para seleção)",
        "bm_H":"Altura manométrica (m):","bm_Q":"Vazão nominal (m³/h):",
        "bm_P":"Potência estimada (kW):","bm_ef":"Eficiência estimada (%):",
        "bm_btn":"📋 Gerar BOM","bm_exp":"⬇️ Exportar BOM como CSV",
        "bm_ok":"✅ BOM gerada! Clique em 'Exportar BOM como CSV' para baixar.",
        "ct":"Tag","cd":"Descrição","cq":"Quantidade","cu":"Unidade","cs":"Especificação",
    },
    "en": {
        "app_title": "🏭 Calibration Lab Sizing Tool",
        "app_caption": "Parametric Sizing Tool for Closed-Loop Oil Meter Calibration Laboratory",
        "sidebar_header": "⚙️ Global System Parameters",
        "fluid_sub": "Fluid",
        "manual_cb": "Enter fluid properties manually",
        "density": "Density (kg/m³):",
        "cp": "Specific heat (J/kg·K):",
        "k_lbl": "Thermal conductivity (W/m·K):",
        "mu_lbl": "Dynamic viscosity (Pa·s):",
        "sel_fluid": "Select fluid:",
        "pipe_sub": "Piping (Thermal)",
        "d_in": "Inner diameter (m):",
        "d_out": "Outer diameter (m):",
        "L_p": "Total pipe length (m):",
        "eps_lbl": "Pipe emissivity (ε):",
        "eps_help": "Oxidized carbon steel ≈ 0.7–0.9 | Polished steel ≈ 0.05–0.1",
        "hout_lbl": "External convection coeff. (W/m²·K):",
        "hout_help": "Natural convection in air ≈ 5–15 W/m²·K",
        "tab_th": "🌡️ Thermal Simulation",
        "tab_hy": "💧 System Curve",
        "tab_bm": "📋 Bill of Materials (BOM)",
        "tab_mn": "📖 Manual / Calc. Memo",
        "th_header": "🌡️ Thermal Simulation – Heating & Calibration",
        "sys_data": "System Data",
        "vol": "Total fluid volume in system (m³):",
        "tamb": "Ambient temperature (°C):",
        "mu_nom": "Nominal process viscosity (cP):",
        "tsim": "Total simulation time (h):",
        "heat_ph": "⬇️ Heating Phase",
        "calib_ph": "⬇️ Calibration Phase",
        "p_kw": "Pump nominal power (kW):",
        "p_kw_help": "Power held constant — variation along centrifugal pump curve is small. Engineer chooses max or nominal power.",
        "q_ph": "Phase total flow rate (m³/h):",
        "eff": "Pump efficiency (%):",
        "hf": "Pump heat factor:",
        "hf_help": "Fraction of mechanical power converted to heat in the fluid. Closed-loop centrifugal pump ≈ 1.0.",
        "run": "▶️ Run Thermal Simulation",
        "err_mu": "Could not solve for target viscosity temperatures. Check fluid model and target viscosity.",
        "warn_110": "Calibration start temperature (110% µ) was not reached. Increase simulation time or pump power.",
        "res_hdr": "Results",
        "r_mu": "Target Viscosity",
        "r_Tt": "Target Temp. (100% µ)",
        "r_T90": "Temp. 90% µ",
        "r_T110": "Temp. 110% µ",
        "r_Teq": "Equilibrium Temp.",
        "r_th": "⏱ Heating Time",
        "r_cw": "📏 Calibration Window",
        "na": "Not reached",
        "pl_title": "Fluid Temperature vs Time",
        "pl_x": "Time (h)", "pl_y": "Temperature (°C)",
        "tr_heat": "Heating Phase",
        "tr_calib": "Calibration Phase",
        "cw_lbl": "Calibration Window",
        "hy_header": "💧 System Curve – Hydraulics",
        "hy_pipe": "Piping Parameters",
        "hy_d": "Inner diameter (m):",
        "hy_rug": "Absolute roughness (mm):",
        "hy_rug_h": "New carbon steel ≈ 0.046 mm",
        "hy_L": "Total pipe length (m):",
        "hy_dz": "Geometric elevation / static head (m):",
        "hy_c90": "No. of 90° elbows:",
        "hy_c45": "No. of 45° elbows:",
        "hy_tee": "No. of tees:",
        "hy_ve": "No. of ball valves:",
        "hy_vb": "No. of isolation butterfly valves (K=0.5):",
        "hy_vb_help": "Normally-open butterfly valves used for branch isolation (fixed K=0.5). Do NOT include control valves here — configure them below with variable Kv.",
        "fittings_help": "**Fittings & isolation valves** *(minor losses — K method)*\n\n| Component | K |\n|---|---|\n| 90° elbow | 0.30 |\n| 45° elbow | 0.20 |\n| Tee | 0.50 |\n| Ball valve | 0.10 |\n| Isolation butterfly | 0.50 |",
        "hy_global_note": "ℹ️ Inner diameter, length, roughness, and static head are set in **Global Parameters** (sidebar). Edit there to reflect here.",
        "hy_fittings": "Fittings & Isolation Valves",
        "kv_mode_lbl": "Control valve mode:",
        "kv_mode_single": "Single Kv / linear",
        "kv_mode_curve": "Kv × opening curve (3 points)",
        "kv_curve_help": "Enter 3 points from the manufacturer's curve: opening (%) and corresponding Kv. Log-linear interpolation is used, appropriate for equal-percentage valves.",
        "kv_op_j": "Opening (%)",
        "kv_kv_j": "Kv (m³/h·bar⁰·⁵)",
        "n_ctrl_lbl": "Number of control valves:",
        "n_ctrl_help": "Each control valve has independent Kv and opening. In a closed loop, series valves add their pressure drops directly.",
        "ctrl_series_note": "ℹ️ Control valves are in **parallel** in the loop — their effective Kv values are summed.",
        "ctrl_v": "Control Valve",
        "kv_lbl": "Control valve Kv (m³/h·bar⁰·⁵) — IEC 60534:",
        "kv_help": "0 = no control valve.",
        "op_lbl": "Valve opening (%)",
        "op_help": "100% = fully open.",
        "fl_hy": "Fluid (for hydraulics)",
        "hy_rho": "Density (kg/m³):",
        "hy_mu_h": "Nominal process viscosity (cP):",
        "hy_mu_h_h": "Viscosity at nominal calibration temperature (same as Thermal Simulation tab).",
        "hy_qmax": "Maximum flow rate for curve (m³/h):",
        "hy_qmax_h": "Right limit of the X axis. Use the laboratory maximum flow rate.",
        "pump_curve_hdr": "Pump Curve (5 points — nominal speed)",
        "pump_curve_help": "Enter 5 Q×H points from the manufacturer curve at nominal speed. The software applies affinity laws to calculate curves at other frequencies.",
        "pc_poles": "Motor number of poles:",
        "pc_poles_help": "2 poles → 3600 RPM (60 Hz) | 4 poles → 1800 RPM (60 Hz) | 6 poles → 1200 RPM (60 Hz)",
        "pc_freq": "Nominal frequency of the provided curve (Hz):",
        "pc_freq_help": "Frequency at which the manufacturer measured the 5 points. Usually 60 Hz.",
        "pc_fmin": "VFD minimum frequency (Hz):",
        "pc_fmax": "VFD maximum frequency (Hz):",
        "pc_Q_lbl": "Q{} (m³/h):",
        "pc_H_lbl": "H{} (m):",
        "pc_pt": "Point",
        "op_pt_sys": "Operating Points — Variable Speed",
        "op_Q": "Flow at operating point",
        "op_H": "Head at operating point",
        "sys20": "System — 20% opening",
        "sys100": "System — 100% opening",
        "sys_user": "System — user-defined opening",
        "pump_nom": "Pump — {f} Hz (nominal)",
        "pump_fmin": "Pump — {f} Hz (VFD min)",
        "pump_fmax": "Pump — {f} Hz (VFD max)",
        "calc_btn": "📊 Calculate System Curve",
        "kv_calc": "**Quick Kv calculator:**",
        "kv_Q": "Q (m³/h):",
        "kv_dP": "Max ΔP (bar):",
        "kv_rho": "ρ (kg/m³):",
        "kv_res": "**Minimum Kv: {kv:.0f} m³/h·bar⁰·⁵**",
        "m_H800h": "Total H @ 800 m³/h (hot)",
        "m_H800c": "Total H @ 800 m³/h (cold)",
        "m_V800": "Velocity @ 800 m³/h",
        "cold_warn": "⚠️ **Cold Start:** Re = {re:.0f} at 800 m³/h with cold µ ({mu:.0f} cP) — **{reg}** flow. Check motor power and VFD starting torque.",
        "lam": "laminar", "trans": "transitional",
        "tr_hot": "Total H – Hot (nominal)",
        "tr_cold": "Total H – Cold (start-up)",
        "tr_dist": "Distributed Loss – Hot",
        "tr_loc": "Localized Loss (K)",
        "tr_cv": "Control Valve Loss (Kv)",
        "hy_title": "System Curve – Head vs Flow Rate",
        "hy_px": "Flow Rate (m³/h)", "hy_py": "Head (m)",
        "hy_tbl": "Head Loss at Key Flow Points",
        "cQ":"Flow (m³/h)","cHd":"Distributed H (m)","cHl":"Localized H (m)",
        "cHe":"Static H (m)","cHc":"Ctrl Valve H (m)","cHt":"TOTAL H (m)","cdP":"ΔP (bar)",
        "bm_hdr": "📋 Bill of Materials (BOM)",
        "bm_info": "Fill in the parameters to generate the BOM.",
        "bm_tv": "Tank volume (m³):",
        "bm_DN": "Nominal pipe diameter (mm):",
        "bm_L": "Total piping length (m):",
        "bm_c90":"No. of 90° elbows:","bm_c45":"No. of 45° elbows:","bm_tee":"No. of tees:",
        "bm_ve":"No. of ball valves:","bm_vb":"No. of butterfly valves:",
        "bm_vc":"No. of control valves:","bm_pt":"No. of pressure transmitters (PT):",
        "bm_tt":"No. of temperature transmitters (TT):","bm_ftm":"No. of Master Meters:",
        "bm_ftc":"No. of meters under calibration:",
        "bm_pump": "Pump Data (for selection)",
        "bm_H":"Calculated head (m):","bm_Q":"Nominal flow rate (m³/h):",
        "bm_P":"Estimated power (kW):","bm_ef":"Estimated efficiency (%):",
        "bm_btn":"📋 Generate BOM","bm_exp":"⬇️ Export BOM as CSV",
        "bm_ok":"✅ BOM generated! Click 'Export BOM as CSV' to download.",
        "ct":"Tag","cd":"Description","cq":"Quantity","cu":"Unit","cs":"Specification",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    lang = st.radio("🌐 Idioma / Language", ["pt", "en"],
                    format_func=lambda x: "🇧🇷 Português" if x == "pt" else "🇺🇸 English",
                    horizontal=True)
    S = TR[lang]

    st.header(S["sidebar_header"])
    st.subheader(S["fluid_sub"])

    use_manual = st.checkbox(S["manual_cb"], value=False)
    if use_manual:
        rho      = st.number_input(S["density"], min_value=100.0, value=850.0)
        cp_fluid = st.number_input(S["cp"],      min_value=0.1,   value=2000.0)
        k_fluid  = st.number_input(S["k_lbl"],   min_value=0.01,  value=0.12)
        _mu      = st.number_input(S["mu_lbl"],  min_value=0.001, value=0.025)
        viscosity_model = lambda Tf, m=_mu: m
        fluid_choice = "Manual"
    else:
        FLUIDS = [
            "KRD MAX 225 (11.4 - 40.8 cP)",
            "KRD MAX 2205 (82.5 - 402 cP)",
            "KRD MAX 685 (68.2 - 115.6 cP)",
            "KRD MAX 55 (2.4 - 4.64 cP)",
        ]
        fluid_choice = st.selectbox(S["sel_fluid"], FLUIDS)
        rho = 850.0; cp_fluid = 2000.0; k_fluid = 0.12
        _vm = {
            FLUIDS[0]: lambda Tf: 0.1651  * np.exp(-0.046 * Tf),
            FLUIDS[1]: lambda Tf: 1.9133  * np.exp(-0.053 * Tf),
            FLUIDS[2]: lambda Tf: 0.5933  * np.exp(-0.054 * Tf),
            FLUIDS[3]: lambda Tf: -9e-08*Tf**3 + 1e-05*Tf**2 - 0.0007*Tf + 0.0165,
        }
        viscosity_model = _vm[fluid_choice]

    st.subheader(S["pipe_sub"])
    d_inner  = st.number_input(S["d_in"],    min_value=0.01, value=0.2571)
    D_outer  = st.number_input(S["d_out"],   min_value=0.01, value=0.3238)
    L_pipe   = st.number_input(S["L_p"],     min_value=1.0,  value=40.0)
    rug_mm   = st.number_input(S["hy_rug"],  min_value=0.001, value=0.046, help=S["hy_rug_h"])
    dz_glob  = st.number_input(S["hy_dz"],   value=2.0)
    eps_emit = st.number_input(S["eps_lbl"], min_value=0.01, max_value=1.0,
                               value=0.85, help=S["eps_help"])
    h_ext    = st.number_input(S["hout_lbl"], min_value=1.0, value=10.0, help=S["hout_help"])

    st.divider()
    st.subheader("📄 " + ("Exportar Relatório" if lang=="pt" else "Export Report"))
    pdf_ready = ('th_data' in st.session_state or 'hy_data' in st.session_state)
    if pdf_ready:
        if st.button("⬇️ " + ("Gerar e Baixar PDF" if lang=="pt" else "Generate & Download PDF"),
                     type="primary", key="btn_pdf"):
            global_params = {
                'fluid': fluid_choice, 'd_inner': d_inner, 'D_outer': D_outer,
                'L_pipe': L_pipe, 'rug_mm': rug_mm, 'dz_glob': dz_glob,
                'eps_emit': eps_emit, 'h_ext': h_ext,
            }
            try:
                pdf_bytes = build_report_pdf(
                    lang,
                    st.session_state.get('th_data'),
                    st.session_state.get('hy_data'),
                    global_params,
                )
                st.download_button(
                    label="📥 " + ("Clique para baixar" if lang=="pt" else "Click to download"),
                    data=pdf_bytes,
                    file_name=f"Calibration_Lab_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    key="dl_pdf",
                )
            except Exception as e:
                st.error(f"Erro ao gerar PDF: {e}")
    else:
        st.caption("⚠️ " + ("Execute ao menos uma simulação para habilitar o relatório."
                             if lang=="pt" else
                             "Run at least one simulation to enable the report."))

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title(S["app_title"])
st.caption(S["app_caption"])

# ─────────────────────────────────────────────────────────────────────────────
# TABS (no PID tab)
# ─────────────────────────────────────────────────────────────────────────────
tab_hy, tab_th, tab_bm, tab_mn = st.tabs([
    S["tab_hy"], S["tab_th"], S["tab_bm"], S["tab_mn"]
])

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & HELPERS
# ─────────────────────────────────────────────────────────────────────────────
SIGMA    = 5.670374419e-8
K_STEEL  = 45.0


def solve_visc_temp(visc_fn, mu_pa):
    from scipy.optimize import brentq
    try:
        return brentq(lambda T: visc_fn(T) - mu_pa, 0, 250)
    except Exception:
        return None


def solve_Ts(Tf, Tamb, R_int, R_ext, A_rad, eps, nit=15):
    """Newton-Raphson for outer surface temperature."""
    Tamb_K = Tamb + 273.15
    Ts = Tf
    for _ in range(nit):
        TsK = Ts + 273.15
        F   = (Tf-Ts)/R_int - (Ts-Tamb)/R_ext - eps*SIGMA*A_rad*(TsK**4 - Tamb_K**4)
        dF  = -1/R_int - 1/R_ext - 4*eps*SIGMA*A_rad*TsK**3
        d   = -F/dF; Ts += d
        if abs(d) < 1e-4:
            break
    return Ts


def euler_step(Tf, Tamb, F_flow, W_pump,
               rho, cp, kf, d_in, D_out, L, eps, hext, m_tot):
    """
    Returns dT/dt for one Euler step.
    F_flow [m³/s] affects Re → Nu → h_in → thermal resistance → heat loss.
    W_pump [W] is held constant (user input).
    """
    mu   = float(viscosity_model(Tf))
    Re   = (4 * F_flow * rho) / (math.pi * d_in * mu) if mu > 0 else 1e6
    Pr   = (mu * cp) / kf
    Nu   = 0.023 * max(Re, 1)**0.8 * max(Pr, 0.01)**0.33
    h_in = Nu * kf / d_in
    Rci  = 1.0 / (h_in * math.pi * d_in * L)
    Rcp  = math.log(D_out / d_in) / (2.0 * math.pi * K_STEEL * L)
    Rco  = 1.0 / (hext * math.pi * D_out * L)
    Rint = Rci + Rcp
    Arad = math.pi * D_out * L
    Ts   = solve_Ts(Tf, Tamb, Rint, Rco, Arad, eps)
    Qloss = (Tf - Ts) / Rint
    return (W_pump - Qloss) / (m_tot * cp)


def colebrook(Re, er):
    """
    Fator de atrito de Darcy com interpolação linear na zona de transição (2300 < Re < 4000)
    para evitar a descontinuidade numérica na troca entre Hagen-Poiseuille e Colebrook-White.
    """
    Re = max(Re, 1)
    if Re <= 2300:
        return 64 / Re
    # Colebrook-White turbulento
    f_turb = 0.25 / (math.log10(er/3.7 + 5.74/max(Re,1)**0.9))**2
    for _ in range(50):
        fn = (1/(-2*math.log10(er/3.7 + 2.51/(Re*math.sqrt(f_turb)))))**2
        if abs(fn - f_turb) < 1e-8: break
        f_turb = fn
    if Re >= 4000:
        return f_turb
    # Zona de transição 2300–4000: interpolação linear entre os dois regimes
    f_lam_2300  = 64 / 2300
    f_turb_4000 = 0.25 / (math.log10(er/3.7 + 5.74/4000**0.9))**2
    alpha = (Re - 2300) / (4000 - 2300)
    return f_lam_2300 + alpha * (f_turb_4000 - f_lam_2300)


def head_loss(Q, d, rug, L, dz, c90, c45, tee, ve, vb_bloq, ctrl_valves, rho_f, mu_f):
    """
    ctrl_valves: list of (Kv, opening_pct) tuples — parallel control valves.
    vb_bloq: butterfly isolation valves (normally open, fixed K=0.5).
    """
    if Q <= 0:
        return 0, 0, 0, dz, 0
        
    V   = (Q/3600) / (math.pi*d**2/4)
    Re  = rho_f*V*d/mu_f
    f   = colebrook(Re, rug/d)
    
    # Perdas distribuídas e localizadas baseadas na linha principal
    Hd  = f*(L/d)*V**2/(2*9.81)
    Hl  = (c90*0.3 + c45*0.2 + tee*0.5 + ve*0.1 + vb_bloq*0.5)*V**2/(2*9.81)
    
    # === CÁLCULO DAS VÁLVULAS DE CONTROLE (PARALELO) ===
    Hc = 0.0
    Kv_eq = 0.0
    tem_valvula = False
    
    for Kv, op in ctrl_valves:
        if Kv > 0:
            tem_valvula = True
            if op > 0:
                # Soma os Kvs efetivos das linhas em paralelo que estão abertas
                Kve = Kv * (op / 100.0) 
                Kv_eq += Kve

    if tem_valvula:
        if Kv_eq > 0:
            # Calcula a perda de carga baseada no Kv equivalente do sistema paralelo
            Hc = (Q / Kv_eq)**2 * (rho_f / 1000.0) * 1e5 / (rho_f * 9.81)
        else:
            # Se todas as válvulas estiverem 0% abertas, simula bloqueio total
            Hc = 9999.0 
            
    # Retorna todos os 5 valores esperados pelo script
    return Hd+Hl+dz+Hc, Hd, Hl, dz, Hc


def hm(h):
    return f"{int(h)}h {int((h-int(h))*60)}min"


# ─────────────────────────────────────────────────────────────────────────────
# PDF REPORT BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def build_report_pdf(lang, th_data, hy_data, global_params):
    """
    Build a PDF report in memory and return bytes.
    th_data  : dict with thermal simulation results (or None)
    hy_data  : dict with hydraulic results (or None)
    global_params: dict with system-level inputs
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, HRFlowable,
                                    Image as RLImage, PageBreak)
    import tempfile, os

    PT = lang == "pt"

    # ── Styles ────────────────────────────────────────────────────────────────
    styles = getSampleStyleSheet()
    style_title   = ParagraphStyle('ReportTitle', fontSize=18, fontName='Helvetica-Bold',
                                   spaceAfter=6, alignment=TA_CENTER)
    style_subtitle= ParagraphStyle('Subtitle', fontSize=10, fontName='Helvetica',
                                   textColor=colors.HexColor('#555555'),
                                   spaceAfter=14, alignment=TA_CENTER)
    style_h1      = ParagraphStyle('H1', fontSize=13, fontName='Helvetica-Bold',
                                   spaceBefore=14, spaceAfter=6,
                                   textColor=colors.HexColor('#1a3a5c'))
    style_h2      = ParagraphStyle('H2', fontSize=11, fontName='Helvetica-Bold',
                                   spaceBefore=10, spaceAfter=4,
                                   textColor=colors.HexColor('#2c5f8a'))
    style_body    = ParagraphStyle('Body', fontSize=9, fontName='Helvetica',
                                   spaceAfter=4, leading=14)
    style_small   = ParagraphStyle('Small', fontSize=8, fontName='Helvetica',
                                   textColor=colors.HexColor('#666666'), spaceAfter=2)

    # ── Table style helper ────────────────────────────────────────────────────
    def tbl_style(header_color='#1a3a5c'):
        return TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor(header_color)),
            ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
            ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE',   (0,0), (-1,0), 8),
            ('FONTNAME',   (0,1), (-1,-1), 'Helvetica'),
            ('FONTSIZE',   (0,1), (-1,-1), 8),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f0f4f8')]),
            ('GRID',       (0,0), (-1,-1), 0.3, colors.HexColor('#cccccc')),
            ('ALIGN',      (1,1), (-1,-1), 'CENTER'),
            ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 3),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ])

    # ── Plotly fig → PNG → ReportLab Image (requires kaleido) ─────────────────
    def fig_to_rl_image(fig, width_mm=170, height_mm=90):
        try:
            fig.update_layout(width=1000, height=500, margin=dict(l=40,r=40,t=50,b=40))
            img_bytes = fig.to_image(format='png', scale=2)
            buf = io.BytesIO(img_bytes)
            return RLImage(buf, width=width_mm*mm, height=height_mm*mm)
        except Exception:
            msg = ("[Gráfico indisponível — instale kaleido: pip install kaleido]"
                   if PT else
                   "[Chart unavailable — install kaleido: pip install kaleido]")
            return Paragraph(msg, style_small)

    # ── Document ──────────────────────────────────────────────────────────────
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=20*mm, rightMargin=20*mm,
                            topMargin=20*mm, bottomMargin=20*mm)
    story = []
    W = A4[0] - 40*mm   # usable width

    # ── Cover / Header ────────────────────────────────────────────────────────
    story.append(Paragraph("🏭 Calibration Lab Sizing Tool", style_title))
    caption = ("Relatório de Dimensionamento — Laboratório de Calibração de Medidores de Óleo"
               if PT else
               "Sizing Report — Oil Meter Calibration Laboratory")
    story.append(Paragraph(caption, style_subtitle))
    story.append(Paragraph(
        f"{'Gerado em' if PT else 'Generated'}: {datetime.now().strftime('%d/%m/%Y %H:%M')}",
        style_small))
    story.append(HRFlowable(width='100%', thickness=1.5,
                            color=colors.HexColor('#1a3a5c'), spaceAfter=10))

    # ── 1. Global Parameters ──────────────────────────────────────────────────
    gp = global_params
    story.append(Paragraph("1. " + ("Parâmetros Globais do Sistema" if PT else "Global System Parameters"), style_h1))

    gp_rows = [
        [("Fluido" if PT else "Fluid"), gp['fluid'],
         ("Diâmetro interno" if PT else "Inner diameter"), f"{gp['d_inner']:.4f} m"],
        [("Diâmetro externo" if PT else "Outer diameter"), f"{gp['D_outer']:.4f} m",
         ("Comprimento total" if PT else "Total length"), f"{gp['L_pipe']:.1f} m"],
        [("Rugosidade" if PT else "Roughness"), f"{gp['rug_mm']:.3f} mm",
         ("Desnível estático" if PT else "Static head"), f"{gp['dz_glob']:.1f} m"],
        [("Emissividade" if PT else "Emissivity"), f"{gp['eps_emit']:.2f}",
         ("Convecção externa" if PT else "External conv."), f"{gp['h_ext']:.1f} W/m²·K"],
    ]
    t = Table(gp_rows, colWidths=[W*0.22, W*0.28, W*0.22, W*0.28])
    t.setStyle(TableStyle([
        ('FONTNAME',  (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE',  (0,0), (-1,-1), 8),
        ('FONTNAME',  (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME',  (2,0), (2,-1), 'Helvetica-Bold'),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [colors.white, colors.HexColor('#f0f4f8')]),
        ('GRID', (0,0), (-1,-1), 0.3, colors.HexColor('#cccccc')),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t)

    # ── 2. Thermal Simulation ─────────────────────────────────────────────────
    if th_data:
        story.append(Paragraph("2. " + ("Simulação Térmica" if PT else "Thermal Simulation"), style_h1))

        story.append(Paragraph("2.1 " + ("Parâmetros de Entrada" if PT else "Input Parameters"), style_h2))
        inp_rows = [
            [("Volume total" if PT else "Total volume"), f"{th_data['vol_m3']:.1f} m³",
             ("Temp. ambiente" if PT else "Ambient temp."), f"{th_data['T_amb']:.1f} °C"],
            [("Viscosidade nominal" if PT else "Nominal viscosity"), f"{th_data['mu_nom_cP']:.1f} cP",
             ("Tempo simulação" if PT else "Simulation time"), f"{th_data['t_sim_h']:.1f} h"],
        ]
        ph_rows = [
            [("" if PT else ""),
             ("Fase Aquecimento" if PT else "Heating Phase"),
             ("Fase Calibração" if PT else "Calibration Phase")],
            [("Potência bomba" if PT else "Pump power"),
             f"{th_data['P_heat']:.1f} kW", f"{th_data['P_cal']:.1f} kW"],
            [("Vazão" if PT else "Flow rate"),
             f"{th_data['Q_heat']:.0f} m³/h", f"{th_data['Q_cal']:.0f} m³/h"],
            [("Eficiência" if PT else "Efficiency"),
             f"{th_data['ef_heat']:.0f}%", f"{th_data['ef_cal']:.0f}%"],
        ]
        ti = Table(inp_rows, colWidths=[W*0.25, W*0.25, W*0.25, W*0.25])
        ti.setStyle(tbl_style())
        story.append(ti)
        story.append(Spacer(1, 4))
        tp = Table(ph_rows, colWidths=[W*0.35, W*0.325, W*0.325])
        tp.setStyle(tbl_style())
        story.append(tp)

        story.append(Paragraph("2.2 " + ("Resultados" if PT else "Results"), style_h2))
        res_rows = [
            [("Parâmetro" if PT else "Parameter"), ("Valor" if PT else "Value")],
            [("Viscosidade alvo" if PT else "Target viscosity"),    f"{th_data['mu_nom_cP']:.1f} cP"],
            [("Temperatura alvo (100% µ)" if PT else "Target temp. (100% µ)"),
             f"{th_data['Tnom']:.1f} °C" if th_data['Tnom'] else "N/A"],
            [("T início calibração (110% µ)" if PT else "Calib. start temp (110% µ)"),
             f"{th_data['T110']:.1f} °C"],
            [("T fim calibração (90% µ)" if PT else "Calib. end temp (90% µ)"),
             f"{th_data['T90']:.1f} °C" if th_data['T90'] else "N/A"],
            [("Temperatura de equilíbrio" if PT else "Equilibrium temperature"),
             f"{th_data['T_eq']:.1f} °C"],
            [("Tempo de aquecimento" if PT else "Heating time"),    hm(th_data['t110_h'])],
            [("Janela de calibração" if PT else "Calibration window"),
             hm(th_data['cwin_h']) if th_data['cwin_h'] else ("Não atingida" if PT else "Not reached")],
        ]
        tr = Table(res_rows, colWidths=[W*0.6, W*0.4])
        tr.setStyle(tbl_style())
        story.append(tr)

        story.append(Paragraph("2.3 " + ("Curva de Temperatura" if PT else "Temperature Curve"), style_h2))
        story.append(fig_to_rl_image(th_data['fig'], width_mm=170, height_mm=85))

    # ── 3. Hydraulic / System Curve ───────────────────────────────────────────
    if hy_data:
        story.append(PageBreak())
        story.append(Paragraph("3. " + ("Curva do Sistema e Bomba" if PT else "System & Pump Curve"), style_h1))

        story.append(Paragraph("3.1 " + ("Parâmetros Hidráulicos" if PT else "Hydraulic Parameters"), style_h2))
        hy_inp = [
            [("Fluido — densidade" if PT else "Fluid — density"), f"{hy_data['hy_rho']:.0f} kg/m³",
             ("Viscosidade nominal" if PT else "Nominal viscosity"), f"{hy_data['hy_mu_cP']:.1f} cP"],
            [("Vazão máxima" if PT else "Max flow rate"), f"{hy_data['hy_qmax']:.0f} m³/h",
             ("Frequência nominal" if PT else "Nominal frequency"), f"{hy_data['pc_freq0']:.0f} Hz"],
            [("Freq. mínima inversor" if PT else "VFD min freq."), f"{hy_data['pc_fmin']:.0f} Hz",
             ("Freq. máxima inversor" if PT else "VFD max freq."), f"{hy_data['pc_fmax']:.0f} Hz"],
        ]
        th2 = Table(hy_inp, colWidths=[W*0.28, W*0.22, W*0.28, W*0.22])
        th2.setStyle(tbl_style())
        story.append(th2)

        story.append(Paragraph("3.2 " + ("Curva do Sistema e Pontos de Operação" if PT else "System Curve & Operating Points"), style_h2))
        story.append(fig_to_rl_image(hy_data['fig'], width_mm=170, height_mm=95))

        # Operating point tables
        story.append(Paragraph("3.3 " + ("Pontos de Operação — Rotação Variável" if PT else "Operating Points — Variable Speed"), style_h2))
        freq_labels = [
            f"{hy_data['pc_fmin']:.0f} Hz ({'mín.' if PT else 'min'})",
            f"{hy_data['pc_freq0']:.0f} Hz ({'nominal' if PT else 'rated'})",
            f"{hy_data['pc_fmax']:.0f} Hz ({'máx.' if PT else 'max'})",
        ]
        op_header = [("Abertura" if PT else "Opening"),
                     ("Vazão" if PT else "Flow"),
                     ("Altura" if PT else "Head")]

        op_col_w = W / 3 - 2*mm
        op_tables = []
        for fl, ops_t in zip(freq_labels, [hy_data['ops_fmin'], hy_data['ops_fnom'], hy_data['ops_fmax']]):
            rows_op = [op_header]
            for lbl, (q_op, h_op) in ops_t:
                rows_op.append([
                    lbl,
                    f"{q_op:.0f} m³/h" if q_op is not None else "—",
                    f"{h_op:.1f} m"    if h_op is not None else "—",
                ])
            tbl_op = Table(rows_op, colWidths=[op_col_w*0.3, op_col_w*0.35, op_col_w*0.35])
            tbl_op.setStyle(tbl_style('#2c5f8a'))
            op_tables.append([Paragraph(f"<b>{fl}</b>", style_small), tbl_op])

        # Layout 3 tables side by side using a wrapper table
        wrap = Table([[op_tables[0][0], op_tables[1][0], op_tables[2][0]],
                      [op_tables[0][1], op_tables[1][1], op_tables[2][1]]],
                     colWidths=[W/3, W/3, W/3])
        wrap.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'TOP'),
                                  ('LEFTPADDING', (0,0), (-1,-1), 2),
                                  ('RIGHTPADDING', (0,0), (-1,-1), 2)]))
        story.append(wrap)

    # ── Footer note ───────────────────────────────────────────────────────────
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width='100%', thickness=0.5,
                            color=colors.HexColor('#aaaaaa'), spaceAfter=4))
    footer = ("*Validar com cálculos detalhados antes da especificação final.*"
              if PT else
              "*Always validate with detailed engineering calculations before final specification.*")
    story.append(Paragraph(footer, style_small))

    doc.build(story)
    buf.seek(0)
    return buf.read()



# ─────────────────────────────────────────────────────────────────────────────
with tab_th:
    st.header(S["th_header"])

    # Row 1: System Data (full width, 4 columns)
    st.subheader(S["sys_data"])
    s1, s2, s3, s4 = st.columns(4)
    vol_m3     = s1.number_input(S["vol"],    min_value=0.1, value=10.0)
    T_amb      = s2.number_input(S["tamb"],   value=25.0)
    mu_nom_cP  = s3.number_input(S["mu_nom"], value=25.0, min_value=0.01)
    t_sim_h    = s4.number_input(S["tsim"],   min_value=0.1, value=10.0)

    st.divider()

    # Row 2: Heating | Calibration (side by side, below System Data)
    ph1, ph2 = st.columns(2)
    with ph1:
        st.subheader(S["heat_ph"])
        P_heat  = st.number_input(S["p_kw"],  min_value=0.1, value=69.0, key="P_h", help=S["p_kw_help"])
        Q_heat  = st.number_input(S["q_ph"],  min_value=0.1, value=550.0, key="Q_h")
        ef_heat = st.number_input(S["eff"],   min_value=1.0, max_value=100.0, value=58.0, key="ef_h")
        hf_heat = st.number_input(S["hf"],    min_value=0.0, value=1.0, step=0.05, key="hf_h", help=S["hf_help"])
    with ph2:
        st.subheader(S["calib_ph"])
        P_cal   = st.number_input(S["p_kw"],  min_value=0.1, value=69.0, key="P_c", help=S["p_kw_help"])
        Q_cal   = st.number_input(S["q_ph"],  min_value=0.1, value=550.0, key="Q_c")
        ef_cal  = st.number_input(S["eff"],   min_value=1.0, max_value=100.0, value=58.0, key="ef_c")
        hf_cal  = st.number_input(S["hf"],    min_value=0.0, value=1.0, step=0.05, key="hf_c", help=S["hf_help"])

    st.divider()

    if st.button(S["run"], type="primary"):

        mu_110_pa = mu_nom_cP * 1.1 / 1000.0
        mu_90_pa  = mu_nom_cP * 0.9 / 1000.0
        T110 = solve_visc_temp(viscosity_model, mu_110_pa)
        T90  = solve_visc_temp(viscosity_model, mu_90_pa)
        Tnom = solve_visc_temp(viscosity_model, mu_nom_cP / 1000.0)

        if T110 is None or T90 is None:
            st.error(S["err_mu"]); st.stop()

        # Heating phase
        # W_pump constant; F_flow affects thermal resistance path
        W_heat  = P_heat  * (ef_heat/100) * hf_heat * 1000.0   # W
        F_heat  = Q_heat  / 3600.0                              # m³/s
        m_fluid = vol_m3  * rho

        dt    = 1.0
        t_max = t_sim_h * 3600.0
        time  = np.arange(0, t_max, dt)
        Tf_h  = np.zeros(len(time)); Tf_h[0] = T_amb

        for i in range(1, len(time)):
            dTdt = euler_step(Tf_h[i-1], T_amb, F_heat, W_heat,
                              rho, cp_fluid, k_fluid,
                              d_inner, D_outer, L_pipe,
                              eps_emit, h_ext, m_fluid)
            Tf_h[i] = Tf_h[i-1] + dTdt * dt

        idx110 = np.where(Tf_h >= T110)[0]
        if len(idx110) == 0:
            st.warning(S["warn_110"]); st.stop()

        t110_s  = time[idx110[0]]
        T110_v  = Tf_h[idx110[0]]
        t110_h  = t110_s / 3600.0
        mask_h  = time <= t110_s
        t_hp    = time[mask_h]; T_hp = Tf_h[mask_h]

        # Calibration phase
        W_cal  = P_cal * (ef_cal/100) * hf_cal * 1000.0
        F_cal  = Q_cal / 3600.0

        tc     = np.arange(t110_s, t_max, dt)
        Tf_c   = np.zeros(len(tc)); Tf_c[0] = T110_v

        for i in range(1, len(tc)):
            dTdt = euler_step(Tf_c[i-1], T_amb, F_cal, W_cal,
                              rho, cp_fluid, k_fluid,
                              d_inner, D_outer, L_pipe,
                              eps_emit, h_ext, m_fluid)
            Tf_c[i] = Tf_c[i-1] + dTdt * dt

        T_eq  = Tf_c[-1]
        idx90 = np.where(Tf_c >= T90)[0]
        if len(idx90) > 0:
            t90_s = tc[idx90[0]]; T90_v = Tf_c[idx90[0]]
            t90_h = t90_s / 3600.0; cwin_h = t90_h - t110_h
        else:
            t90_h = T90_v = cwin_h = None

        # ── Results row 1: viscosities & temperatures (5 metrics) ──
        st.subheader(S["res_hdr"])
        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric(S["r_mu"],  f"{mu_nom_cP:.1f} cP")
        r2.metric(S["r_Tt"],  f"{Tnom:.1f} °C" if Tnom else "N/A")
        r3.metric(S["r_T90"], f"{T90:.1f} °C"  if T90  else "N/A")
        r4.metric(S["r_T110"],f"{T110:.1f} °C")
        r5.metric(S["r_Teq"], f"{T_eq:.1f} °C")

        # ── Results row 2: timing (2 metrics) ──
        t1, t2 = st.columns(2)
        t1.metric(S["r_th"], hm(t110_h))
        t2.metric(S["r_cw"], hm(cwin_h) if cwin_h is not None else S["na"])

        # ── Plot ──
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t_hp/3600, y=T_hp, mode='lines',
            name=S["tr_heat"], line=dict(color='orangered', width=2.5)))
        fig.add_trace(go.Scatter(x=tc/3600, y=Tf_c, mode='lines',
            name=S["tr_calib"], line=dict(color='royalblue', width=2.5)))

        xf = [0, t_sim_h]
        fig.add_trace(go.Scatter(x=xf, y=[T_eq,T_eq], mode='lines',
            name=f'T_eq={T_eq:.1f}°C', line=dict(color='orangered', dash='dash')))
        fig.add_trace(go.Scatter(x=xf, y=[T110,T110], mode='lines',
            name=f'T 110%µ={T110:.1f}°C', line=dict(color='purple', dash='dot')))
        fig.add_trace(go.Scatter(x=[t110_h,t110_h], y=[T_amb-5, T_eq+10], mode='lines',
            name=f't₁₁₀={hm(t110_h)}', line=dict(color='purple', dash='dot')))
        fig.add_trace(go.Scatter(x=[t110_h], y=[T110_v], mode='markers',
            marker=dict(color='purple', size=9), showlegend=False))

        if T90 is not None:
            fig.add_trace(go.Scatter(x=xf, y=[T90,T90], mode='lines',
                name=f'T 90%µ={T90:.1f}°C', line=dict(color='green', dash='dot')))
        if t90_h is not None:
            fig.add_trace(go.Scatter(x=[t90_h,t90_h], y=[T_amb-5, T_eq+10], mode='lines',
                name=f't₉₀={hm(t90_h)}', line=dict(color='green', dash='dot')))
            fig.add_trace(go.Scatter(x=[t90_h], y=[T90_v], mode='markers',
                marker=dict(color='green', size=9), showlegend=False))
            fig.add_vrect(x0=t110_h, x1=t90_h, fillcolor="green", opacity=0.07,
                          layer="below", line_width=0,
                          annotation_text=S["cw_lbl"], annotation_position="top left")

        fig.update_layout(
            title=S["pl_title"], xaxis_title=S["pl_x"], yaxis_title=S["pl_y"],
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
            hovermode="x unified", template="plotly_white", height=520)
        st.plotly_chart(fig, use_container_width=True)

        # Store results for PDF export
        st.session_state['th_data'] = {
            'vol_m3': vol_m3, 'T_amb': T_amb, 'mu_nom_cP': mu_nom_cP,
            't_sim_h': t_sim_h, 'P_heat': P_heat, 'Q_heat': Q_heat,
            'ef_heat': ef_heat, 'P_cal': P_cal, 'Q_cal': Q_cal, 'ef_cal': ef_cal,
            'Tnom': Tnom, 'T90': T90, 'T110': T110, 'T_eq': T_eq,
            't110_h': t110_h, 'cwin_h': cwin_h,
            'fig': fig,
        }


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 – HYDRAULIC SYSTEM CURVE
# ─────────────────────────────────────────────────────────────────────────────
with tab_hy:
    st.header(S["hy_header"])
    st.subheader(S["hy_pipe"])
    st.info(S["hy_global_note"])

    # Show current global pipe params (read-only reference)
    pg1, pg2, pg3, pg4, pg5 = st.columns(5)
    pg1.metric(S["d_in"],  f"{d_inner:.4f} m")
    pg2.metric(S["d_out"], f"{D_outer:.4f} m")
    pg3.metric(S["L_p"],   f"{L_pipe:.1f} m")
    pg4.metric(S["hy_rug"], f"{rug_mm:.3f} mm")
    pg5.metric(S["hy_dz"],  f"{dz_glob:.1f} m")

    # Use global values directly
    hy_d  = d_inner
    hy_rg = rug_mm / 1000.0
    hy_L  = L_pipe
    hy_dz = dz_glob

    st.subheader(S["hy_fittings"])
    hc2_a, hc2_b = st.columns(2)
    with hc2_a:
        hy_c90 = st.number_input(S["hy_c90"], min_value=0, value=4, step=1)
        hy_c45 = st.number_input(S["hy_c45"], min_value=0, value=2, step=1)
    with hc2_b:
        hy_tee = st.number_input(S["hy_tee"], min_value=0, value=2, step=1)
        hy_ve  = st.number_input(S["hy_ve"],  min_value=0, value=3, step=1)
        hy_vb_bloq = st.number_input(S["hy_vb"], min_value=0, value=0, step=1,
                                     help=S["hy_vb_help"])

    st.subheader(S["ctrl_v"])
    with st.expander("ℹ️ Kv — IEC 60534", expanded=False):
        st.markdown(r"""
$$K_v = \frac{Q\,[\text{m}^3/\text{h}]}{\sqrt{\Delta P\,[\text{bar}]\cdot\dfrac{\rho}{1000}}}$$
> $K_v \approx C_v \times 0{,}865$
        """)
        st.markdown(S["kv_calc"])
        kk1, kk2, kk3 = st.columns(3)
        kv_Q_  = kk1.number_input(S["kv_Q"],  value=800.0, key="kvQ")
        kv_dP_ = kk2.number_input(S["kv_dP"], value=1.0, min_value=0.01, key="kvdP")
        kv_r_  = kk3.number_input(S["kv_rho"], value=850.0, key="kvrho")
        st.success(S["kv_res"].format(kv=kv_Q_/math.sqrt(kv_dP_*kv_r_/1000)))

    n_ctrl = st.number_input(S["n_ctrl_lbl"], min_value=0, value=1, step=1,
                             help=S["n_ctrl_help"])
    st.info(S["ctrl_series_note"])

    # Build ctrl_valves list AND store FCV curve data per valve
    ctrl_valves    = []   # list of (Kv_effective_at_user_opening) for head_loss
    fcv_curve_data = []   # list of (mode, interp_fn) for sys_curve override

    for i in range(int(n_ctrl)):
        st.markdown(f"**{S['ctrl_v']} {i+1}**")
        cv_mode = st.radio(S["kv_mode_lbl"], [S["kv_mode_single"], S["kv_mode_curve"]],
                           key=f"cvmode_{i}", horizontal=True)

        if cv_mode == S["kv_mode_single"]:
            cv1, cv2 = st.columns(2)
            kv_i = cv1.number_input(S["kv_lbl"], min_value=0.0, value=870.0,
                                    key=f"kv_{i}", help=S["kv_help"])
            op_i = cv2.slider(S["op_lbl"], 0, 100, 100,
                              key=f"op_{i}", help=S["op_help"])
            ctrl_valves.append((kv_i, op_i))
            fcv_curve_data.append(("linear", kv_i))

        else:  # 3-point curve mode
            st.caption(S["kv_curve_help"])
            default_cv_pts = [(25, 45), (50, 394), (100, 913)]
            cv_pts_op = []; cv_pts_kv = []
            cc = st.columns(3)
            for j, (dop, dkv) in enumerate(default_cv_pts):
                with cc[j]:
                    st.markdown(f"**{S['pc_pt']} {j+1}**")
                    op_j = st.number_input(S["kv_op_j"], value=float(dop),
                                           min_value=0.0, max_value=100.0,
                                           key=f"cvop_{i}_{j}")
                    kv_j = st.number_input(S["kv_kv_j"], value=float(dkv),
                                           min_value=0.0, key=f"cvkv_{i}_{j}")
                    cv_pts_op.append(op_j); cv_pts_kv.append(kv_j)

            op_user_i = st.slider(S["op_lbl"], 0, 100, 100,
                                  key=f"op_{i}", help=S["op_help"])

            # Build interpolator (log-linear on Kv vs opening — equal % behaviour)
            import numpy as _np
            from scipy.interpolate import interp1d as _interp1d
            _ops = _np.array(cv_pts_op, dtype=float)
            _kvs = _np.array(cv_pts_kv, dtype=float)
            sort_i = _np.argsort(_ops)
            _ops, _kvs = _ops[sort_i], _kvs[sort_i]
            # log-linear interpolation is natural for equal-percentage valves
            _log_kv = _np.log(_kvs)
            _interp = _interp1d(_ops, _log_kv, kind='linear',
                                fill_value=(_log_kv[0], _log_kv[-1]), bounds_error=False)
            kv_at_user = float(_np.exp(_interp(op_user_i)))
            ctrl_valves.append((kv_at_user, 100))   # Kv already resolved; pass 100% open
            fcv_curve_data.append(("curve", (_ops, _kvs, _interp, op_user_i)))

    st.subheader(S["fl_hy"])
    fc1, fc2, fc3 = st.columns(3)
    hy_rho    = fc1.number_input(S["hy_rho"],  min_value=100.0, value=850.0)
    hy_mu_cP  = fc2.number_input(S["hy_mu_h"], min_value=0.01, value=25.0, help=S["hy_mu_h_h"])
    hy_mu_hot = hy_mu_cP / 1000.0   # convert to Pa·s for calculations
    hy_qmax   = fc3.number_input(S["hy_qmax"], min_value=50.0, value=900.0, help=S["hy_qmax_h"])

    # ── Pump curve inputs ──────────────────────────────────────────────────────
    st.subheader(S["pump_curve_hdr"])
    st.caption(S["pump_curve_help"])

    pc1, pc2, pc3 = st.columns(3)
    pc_poles = pc1.selectbox(S["pc_poles"], [2, 4, 6, 8], index=1, help=S["pc_poles_help"])
    pc_freq0 = pc2.number_input(S["pc_freq"], min_value=10.0, value=60.0, help=S["pc_freq_help"])
    pc_fmin  = pc3.number_input(S["pc_fmin"], min_value=5.0,  value=20.0)
    pc_fmax  = pc3.number_input(S["pc_fmax"], min_value=10.0, value=60.0)

    # Default plausible 5-point pump curve (Q in m³/h, H in m)
    default_pts = [(0, 45), (200, 42), (400, 35), (600, 22), (800, 5)]
    pump_pts = []
    col_headers = st.columns(5)
    for i, (dq, dh) in enumerate(default_pts):
        with col_headers[i]:
            st.markdown(f"**{S['pc_pt']} {i+1}**")
            q_i = st.number_input(S["pc_Q_lbl"].format(i+1), value=float(dq),
                                  min_value=0.0, key=f"pcQ{i}")
            h_i = st.number_input(S["pc_H_lbl"].format(i+1), value=float(dh),
                                  min_value=0.0, key=f"pcH{i}")
            pump_pts.append((q_i, h_i))

    if st.button(S["calc_btn"], type="primary"):
        from scipy.interpolate import CubicSpline
        from scipy.optimize import brentq

        Qr = np.linspace(0, hy_qmax, 400)

        # ── System curves at 3 valve openings ─────────────────────────────────
        import numpy as _np2
        def resolve_ctrl_valves(op_pct):
            """Build ctrl_valves list with Kv resolved at op_pct for each valve."""
            cv_resolved = []
            for idx, (mode, data) in enumerate(fcv_curve_data):
                if mode == "linear":
                    kv_nom = data
                    cv_resolved.append((kv_nom, op_pct))
                else:
                    _ops_c, _kvs_c, _interp_c, _user_op_c = data
                    kv_resolved = float(_np2.exp(_interp_c(op_pct)))
                    cv_resolved.append((kv_resolved, 100))  # Kv already at this opening
            return cv_resolved

        def sys_curve(Q_arr, op_pct):
            cv_ov = resolve_ctrl_valves(op_pct)
            H = []
            for Q in Q_arr:
                h, *_ = head_loss(Q, hy_d, hy_rg, hy_L, hy_dz,
                                  hy_c90, hy_c45, hy_tee, hy_ve, hy_vb_bloq,
                                  cv_ov, hy_rho, hy_mu_hot)
                H.append(h)
            return _np2.array(H)

        H_sys20   = sys_curve(Qr, 20)
        H_sys100  = sys_curve(Qr, 100)
        # user_op: for curve mode, read the real slider value stored in fcv_curve_data
        if fcv_curve_data:
            mode0, data0 = fcv_curve_data[0]
            if mode0 == "linear":
                user_op = ctrl_valves[0][1] if ctrl_valves else 100
            else:
                user_op = int(data0[3])  # op_user_i stored as 4th element
        else:
            user_op = 100
        H_sys_usr = sys_curve(Qr, user_op)

        # ── Pump curves via affinity laws ──────────────────────────────────────
        # Fit cubic spline on nominal curve
        Qp = np.array([p[0] for p in pump_pts])
        Hp = np.array([p[1] for p in pump_pts])
        # sort by Q
        sort_idx = np.argsort(Qp)
        Qp, Hp = Qp[sort_idx], Hp[sort_idx]
        cs_pump = CubicSpline(Qp, Hp, extrapolate=False)

        def pump_curve_at_freq(f_target):
            ratio = f_target / pc_freq0
            Q_sc = Qp * ratio          # Q scales with speed ratio
            H_sc = Hp * ratio**2       # H scales with speed ratio squared
            cs = CubicSpline(Q_sc, H_sc, extrapolate=False)
            Q_full = np.linspace(0, Q_sc[-1], 400)
            H_full = cs(Q_full)
            # clip negatives
            H_full = np.where(H_full < 0, np.nan, H_full)
            return Q_full, H_full, cs, Q_sc[-1]

        Qnom, Hnom, cs_nom, Qmax_nom = pump_curve_at_freq(pc_freq0)
        Qmin, Hmin, cs_fmin, Qmax_min = pump_curve_at_freq(pc_fmin)
        Qmx,  Hmx,  cs_fmax, Qmax_mx  = pump_curve_at_freq(pc_fmax)

        # ── Operating points: intersection pump(nominal) × each system curve ──
        def find_op(cs_pump_f, Qmax_f, sys_H_arr, sys_Q_arr=Qr):
            """Find Q where pump_H(Q) = sys_H(Q) via brentq."""
            cs_sys = CubicSpline(sys_Q_arr, sys_H_arr, extrapolate=True)
            diff = lambda Q: float(cs_pump_f(Q)) - float(cs_sys(Q))
            # search within valid pump range
            Q_search = np.linspace(1, min(Qmax_f, hy_qmax), 300)
            d_vals = [diff(q) for q in Q_search]
            # find sign changes
            ops = []
            for j in range(len(d_vals)-1):
                if d_vals[j] * d_vals[j+1] < 0:
                    try:
                        q_op = brentq(diff, Q_search[j], Q_search[j+1])
                        h_op = float(cs_sys(q_op))
                        ops.append((q_op, h_op))
                    except Exception:
                        pass
            return ops[-1] if ops else (None, None)  # take rightmost intersection

        op20   = find_op(cs_nom, Qmax_nom, H_sys20)
        op100  = find_op(cs_nom, Qmax_nom, H_sys100)
        op_usr = find_op(cs_nom, Qmax_nom, H_sys_usr)

        # ── Plot ──────────────────────────────────────────────────────────────
        fh = go.Figure()

        # System curves — sólida para 20% e 100%, pontilhada para abertura do usuário
        fh.add_trace(go.Scatter(x=Qr, y=H_sys20, mode='lines',
            name=S["sys20"], line=dict(color='steelblue', width=2.5)))
        fh.add_trace(go.Scatter(x=Qr, y=H_sys100, mode='lines',
            name=S["sys100"], line=dict(color='royalblue', width=2.5)))
        if user_op not in (20, 100):
            fh.add_trace(go.Scatter(x=Qr, y=H_sys_usr, mode='lines',
                name=S["sys_user"] + f" ({user_op}%)",
                line=dict(color='cornflowerblue', width=2, dash='dot')))

        # Pump curves: fmin=amarelo, fmax=vermelho, nominal=pontilhado laranja
        fh.add_trace(go.Scatter(x=Qmin, y=Hmin, mode='lines',
            name=S["pump_fmin"].format(f=pc_fmin),
            line=dict(color='#FFB300', width=2.5)))          # amarelo âmbar
        fh.add_trace(go.Scatter(x=Qmx, y=Hmx, mode='lines',
            name=S["pump_fmax"].format(f=pc_fmax),
            line=dict(color='#CC0000', width=2.5)))          # vermelho
        if pc_freq0 not in (pc_fmin, pc_fmax):
            fh.add_trace(go.Scatter(x=Qnom, y=Hnom, mode='lines',
                name=S["pump_nom"].format(f=pc_freq0),
                line=dict(color='orangered', width=2, dash='dot')))

        # ── Operating points para as 3 rotações ──────────────────────────────
        # Calcular OPs para fmin e fmax (nominal já calculado acima)
        def ops_for_freq(cs_f, Qmax_f):
            o20_  = find_op(cs_f, Qmax_f, H_sys20)
            ousr_ = find_op(cs_f, Qmax_f, H_sys_usr)
            o100_ = find_op(cs_f, Qmax_f, H_sys100)
            return o20_, ousr_, o100_

        ops_fmin = ops_for_freq(cs_fmin, Qmax_min)
        ops_fnom = (op20, op_usr, op100)
        ops_fmax = ops_for_freq(cs_fmax, Qmax_mx)

        # Cores por abertura da válvula (consistente entre rotações)
        op_valve_colors = {
            '20%':          '#e67e00',   # laranja
            f'{user_op}%':  '#8B008B',   # roxo escuro
            '100%':         '#006400',   # verde escuro
        }
        # Símbolos por rotação: mín=triângulo-baixo, nominal=círculo, máx=triângulo-cima
        op_speed_symbols = {
            'fmin': ('triangle-down', pc_fmin),
            'fnom': ('circle',        pc_freq0),
            'fmax': ('triangle-up',   pc_fmax),
        }

        for speed_key, ops_tuple in [('fmin', ops_fmin), ('fnom', ops_fnom), ('fmax', ops_fmax)]:
            sym, freq_val = op_speed_symbols[speed_key]
            for valve_lbl, (q_op, h_op) in [
                ('20%',         ops_tuple[0]),
                (f'{user_op}%', ops_tuple[1]),
                ('100%',        ops_tuple[2]),
            ]:
                if q_op is not None:
                    col = op_valve_colors[valve_lbl]
                    fh.add_trace(go.Scatter(
                        x=[q_op], y=[h_op], mode='markers+text',
                        marker=dict(color=col, size=13, symbol=sym,
                                    line=dict(color='white', width=1.5)),
                        text=[f"  <b>{valve_lbl} @ {freq_val}Hz</b><br>"
                              f"  Q={q_op:.0f} m³/h | H={h_op:.1f} m"],
                        textfont=dict(color=col, size=11),
                        textposition='middle right',
                        showlegend=False
                    ))

        fh.add_vline(x=hy_qmax, line_dash="dot", line_color="gray",
                     annotation_text=f"Q={hy_qmax:.0f} m³/h", annotation_position="top right")

        # Y-axis: limit to 15% above highest pump shut-off head
        all_pump_H = list(Hnom[~np.isnan(Hnom)]) + list(Hmin[~np.isnan(Hmin)]) + list(Hmx[~np.isnan(Hmx)])
        y_max = max(all_pump_H) * 1.15 if all_pump_H else None

        fh.update_layout(
            title=S["hy_title"], xaxis_title=S["hy_px"], yaxis_title=S["hy_py"],
            xaxis=dict(range=[0, hy_qmax * 1.05]),
            yaxis=dict(range=[0, y_max]),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
            template="plotly_white", hovermode="x unified", height=560)
        st.plotly_chart(fh, use_container_width=True)

        # ── Operating point tables: 3 colunas — rotação mín / nominal / máx ────
        if any(q is not None for q, _ in [op20, op100, op_usr]):
            st.subheader(S["op_pt_sys"])

            def make_op_df(ops_tuple):
                o20_, ousr_, o100_ = ops_tuple
                rows_ = []
                for lbl, (q_op, h_op) in [
                    ("20%",         o20_),
                    (f"{user_op}%", ousr_),
                    ("100%",        o100_),
                ]:
                    rows_.append({
                        S["op_lbl"]: lbl,
                        S["op_Q"]:   f"{q_op:.1f} m³/h" if q_op is not None else "—",
                        S["op_H"]:   f"{h_op:.1f} m"    if h_op is not None else "—",
                    })
                return pd.DataFrame(rows_)

            tc1, tc2, tc3 = st.columns(3)
            with tc1:
                st.markdown(f"**{S['pump_fmin'].format(f=pc_fmin)}**")
                st.dataframe(make_op_df(ops_fmin), use_container_width=True, hide_index=True)
            with tc2:
                st.markdown(f"**{S['pump_nom'].format(f=pc_freq0)}**")
                st.dataframe(make_op_df(ops_fnom), use_container_width=True, hide_index=True)
            with tc3:
                st.markdown(f"**{S['pump_fmax'].format(f=pc_fmax)}**")
                st.dataframe(make_op_df(ops_fmax), use_container_width=True, hide_index=True)

        # Store results for PDF export
        def ops_as_list(ops_tuple):
            o20_, ousr_, o100_ = ops_tuple
            return [("20%", o20_), (f"{user_op}%", ousr_), ("100%", o100_)]

        st.session_state['hy_data'] = {
            'hy_rho': hy_rho, 'hy_mu_cP': hy_mu_cP, 'hy_qmax': hy_qmax,
            'pc_freq0': pc_freq0, 'pc_fmin': pc_fmin, 'pc_fmax': pc_fmax,
            'ops_fmin': ops_as_list(ops_fmin),
            'ops_fnom': ops_as_list(ops_fnom),
            'ops_fmax': ops_as_list(ops_fmax),
            'fig': fh,
        }


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 – BOM
# ─────────────────────────────────────────────────────────────────────────────
with tab_bm:
    st.header(S["bm_hdr"])
    st.info(S["bm_info"])
    bc1, bc2 = st.columns(2)
    with bc1:
        bv = st.number_input(S["bm_tv"],  min_value=0.1, value=10.0,  key="bm_tv")
        bDN= st.number_input(S["bm_DN"],  value=250, step=25,          key="bm_DN")
        bL = st.number_input(S["bm_L"],   min_value=1.0, value=40.0,  key="bm_L")
        bc90=st.number_input(S["bm_c90"], min_value=0, value=4,  step=1, key="bm_c90")
        bc45=st.number_input(S["bm_c45"], min_value=0, value=2,  step=1, key="bm_c45")
        bt  =st.number_input(S["bm_tee"], min_value=0, value=2,  step=1, key="bm_tee")
    with bc2:
        bve =st.number_input(S["bm_ve"],  min_value=0, value=4,  step=1, key="bm_ve")
        bvb =st.number_input(S["bm_vb"],  min_value=0, value=1,  step=1, key="bm_vb")
        bvc =st.number_input(S["bm_vc"],  min_value=0, value=1,  step=1, key="bm_vc")
        bpt =st.number_input(S["bm_pt"],  min_value=0, value=4,  step=1, key="bm_pt")
        btt =st.number_input(S["bm_tt"],  min_value=0, value=3,  step=1, key="bm_tt")
        bftm=st.number_input(S["bm_ftm"], min_value=1, value=1,  step=1, key="bm_ftm")
        bftc=st.number_input(S["bm_ftc"], min_value=1, value=1,  step=1, key="bm_ftc")

    st.subheader(S["bm_pump"])
    bp1, bp2 = st.columns(2)
    bH  = bp1.number_input(S["bm_H"],  min_value=0.0, value=30.0,  key="bm_H")
    bQ  = bp1.number_input(S["bm_Q"],  min_value=0.0, value=800.0, key="bm_Q")
    bP  = bp2.number_input(S["bm_P"],  min_value=0.0, value=90.0,  key="bm_P")
    bef = bp2.number_input(S["bm_ef"], min_value=1.0, max_value=100.0, value=70.0, key="bm_ef")

    if st.button(S["bm_btn"], type="primary"):
        CT=S["ct"]; CD=S["cd"]; CQ=S["cq"]; CU=S["cu"]; CS=S["cs"]
        if lang == "pt":
            rows_bom = [
                {CT:"TQ-001",    CD:"Tanque de processo",               CQ:1,    CU:"un", CS:f"{bv:.1f} m³"},
                {CT:"PU-001",    CD:"Bomba centrífuga",                 CQ:1,    CU:"un", CS:f"Q={bQ:.0f}m³/h|H={bH:.1f}m|P≈{bP:.0f}kW|η≈{bef:.0f}%"},
                {CT:"IHF-001",   CD:"Inversor de frequência",           CQ:1,    CU:"un", CS:f"P≈{bP:.0f} kW"},
                {CT:"TUB-001",   CD:f"Tubulação AC DN{bDN}",           CQ:bL,   CU:"m",  CS:f"DN{bDN}mm, tubo nu"},
                {CT:"CUR90-XXX", CD:f"Curva 90° DN{bDN}",              CQ:bc90, CU:"un", CS:f"DN{bDN}mm, solda"},
                {CT:"CUR45-XXX", CD:f"Curva 45° DN{bDN}",              CQ:bc45, CU:"un", CS:f"DN{bDN}mm, solda"},
                {CT:"TE-XXX",    CD:f"Tê DN{bDN}",                     CQ:bt,   CU:"un", CS:f"DN{bDN}mm"},
                {CT:"VBL-XXX",   CD:"Válvula esfera (bloqueio)",        CQ:bve,  CU:"un", CS:f"DN{bDN}mm"},
                {CT:"VBB-XXX",   CD:"Válvula borboleta",                CQ:bvb,  CU:"un", CS:f"DN{bDN}mm"},
                {CT:"VFC-XXX",   CD:"Válvula de controle c/ atuador",  CQ:bvc,  CU:"un", CS:f"DN{bDN}mm"},
                {CT:"PT-XXX",    CD:"Transmissor de pressão",          CQ:bpt,  CU:"un", CS:"4-20mA, HART"},
                {CT:"TT-XXX",    CD:"Transmissor de temperatura",      CQ:btt,  CU:"un", CS:"4-20mA, HART"},
                {CT:"FT-MASTER", CD:"Master Meter (referência)",       CQ:bftm, CU:"un", CS:"Coriolis/ultrassônico, classe 0,02%"},
                {CT:"FT-CALIB",  CD:"Medidor em calibração (UUT)",     CQ:bftc, CU:"un", CS:"A definir"},
            ]
        else:
            rows_bom = [
                {CT:"TQ-001",    CD:"Process tank",                CQ:1,    CU:"ea", CS:f"{bv:.1f} m³"},
                {CT:"PU-001",    CD:"Centrifugal pump",            CQ:1,    CU:"ea", CS:f"Q={bQ:.0f}m³/h|H={bH:.1f}m|P≈{bP:.0f}kW|η≈{bef:.0f}%"},
                {CT:"VFD-001",   CD:"Variable frequency drive",    CQ:1,    CU:"ea", CS:f"P≈{bP:.0f} kW"},
                {CT:"PIP-001",   CD:f"CS pipe DN{bDN}",            CQ:bL,   CU:"m",  CS:f"DN{bDN}mm, bare"},
                {CT:"ELB90-XXX", CD:f"90° elbow DN{bDN}",          CQ:bc90, CU:"ea", CS:f"DN{bDN}mm, BW"},
                {CT:"ELB45-XXX", CD:f"45° elbow DN{bDN}",          CQ:bc45, CU:"ea", CS:f"DN{bDN}mm, BW"},
                {CT:"TEE-XXX",   CD:f"Tee DN{bDN}",                CQ:bt,   CU:"ea", CS:f"DN{bDN}mm"},
                {CT:"BV-XXX",    CD:"Ball valve (isolation)",       CQ:bve,  CU:"ea", CS:f"DN{bDN}mm"},
                {CT:"BFV-XXX",   CD:"Butterfly valve",              CQ:bvb,  CU:"ea", CS:f"DN{bDN}mm"},
                {CT:"CV-XXX",    CD:"Control valve w/ actuator",   CQ:bvc,  CU:"ea", CS:f"DN{bDN}mm"},
                {CT:"PT-XXX",    CD:"Pressure transmitter",         CQ:bpt,  CU:"ea", CS:"4-20mA, HART"},
                {CT:"TT-XXX",    CD:"Temperature transmitter",      CQ:btt,  CU:"ea", CS:"4-20mA, HART"},
                {CT:"FT-MASTER", CD:"Master Meter (reference)",     CQ:bftm, CU:"ea", CS:"Coriolis/ultrasonic, class 0.02%"},
                {CT:"FT-CALIB",  CD:"Meter under calibration (UUT)",CQ:bftc, CU:"ea", CS:"TBD"},
            ]
        df_bom = pd.DataFrame(rows_bom)
        st.dataframe(df_bom, use_container_width=True, hide_index=True)
        buf = io.StringIO()
        df_bom.to_csv(buf, index=False, sep=";", encoding="utf-8-sig")
        st.download_button(label=S["bm_exp"],
                           data=buf.getvalue().encode("utf-8-sig"),
                           file_name="BOM_Calibration_Lab.csv", mime="text/csv")
        st.success(S["bm_ok"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 – MANUAL
# ─────────────────────────────────────────────────────────────────────────────
with tab_mn:
    if lang == "pt":
        st.markdown(r"""
# 📖 Manual do Usuário e Memorial de Cálculo

## 1. Visão Geral

Ferramenta para dimensionamento paramétrico de laboratório de calibração de medidores de óleo em loop fechado
(1 tanque, 1 bomba centrífuga + inversor de frequência, tubulação descoberta, 0–800 m³/h).

| Aba | Função |
|-----|--------|
| 🌡️ Simulação Térmica | Aquecimento e janela de calibração |
| 💧 Curva do Sistema | Perdas de carga, curvas H×Q quente e fria |
| 📋 BOM | Lista de materiais exportável em CSV |

---

## 2. Simulação Térmica

### 2.1 Potência da Bomba — Premissa

A potência é mantida **constante** (entrada direta em kW). A variação de potência ao longo da curva
de uma bomba centrífuga é pequena e fica a critério do engenheiro usar a potência máxima (conservador)
ou a potência nominal.

### 2.2 Efeito da Vazão na Curva de Temperatura

A vazão de cada fase afeta diretamente a curva de temperatura pelo seguinte caminho:

$$Q_{liq} \uparrow \;\Rightarrow\; Re \uparrow \;\Rightarrow\; Nu \uparrow \;\Rightarrow\; h_{int} \uparrow \;\Rightarrow\; R_{conv,int} \downarrow \;\Rightarrow\; Q_{perda} \uparrow \;\Rightarrow\; \frac{dT}{dt} \downarrow$$

Ou seja: **maior vazão → maior perda de calor → aquecimento mais lento**.
Fases com vazões diferentes produzirão curvas de temperatura com inclinações distintas.

### 2.3 Balanço Térmico — Euler Explícito (Δt = 1 s)

$$m \cdot c_p \cdot \frac{dT_f}{dt} = \dot{W}_p - Q_{perda}$$

- $\dot{W}_p = P_{bomba} \cdot \eta \cdot f_{calor}$ [W] — constante por fase
- $m = V_{total} \cdot \rho$ [kg]

### 2.4 Resistências Térmicas

$$R_{int} = R_{conv,int} + R_{cond,parede}$$

**Convecção interna (Dittus-Boelter):**
$$Nu = 0{,}023 \cdot Re^{0{,}8} \cdot Pr^{0{,}33}, \quad Re = \frac{4 \dot{m}}{\pi d_i \mu}$$

**Condução na parede:** $R_{cond} = \ln(D_e/d_i)\,/\,(2\pi k_{aço} L)$, com $k_{aço}=45$ W/m·K

**Convecção externa:** $R_{conv,ext} = 1\,/\,(h_{ext}\pi D_e L)$

### 2.5 Temperatura da Superfície — Newton-Raphson

$$\frac{T_f - T_s}{R_{int}} = \frac{T_s - T_{amb}}{R_{conv,ext}} + \varepsilon\sigma A_{ext}(T_s^4 - T_{amb}^4)$$

Resolvida iterativamente; perda usada: $Q_{perda} = (T_f - T_s)/R_{int}$

### 2.6 Condição de Calibração

| Evento | Critério |
|--------|----------|
| Início calibração | $\mu(T_{110\%}) = 1{,}10\,\mu_{nom}$ |
| Fim calibração | $\mu(T_{90\%}) = 0{,}90\,\mu_{nom}$ |
| Janela | $\Delta t = t_{90\%} - t_{110\%}$ |

---

## 3. Curva do Sistema

### 3.1 Darcy-Weisbach + Colebrook-White

$$h_f = f \cdot \frac{L}{d_i} \cdot \frac{V^2}{2g}; \quad \frac{1}{\sqrt{f}}=-2\log_{10}\!\left(\frac{\varepsilon_r}{3{,}7}+\frac{2{,}51}{Re\sqrt{f}}\right)$$

Para $Re<2300$: $f=64/Re$.

### 3.2 Perdas Localizadas (K)

| Componente | K |
|---|---|
| Curva 90° | 0,30 |
| Curva 45° | 0,20 |
| Tê | 0,50 |
| Válvula esfera (bloqueio) | 0,10 |
| Válvula borboleta (bloqueio) | 0,50 |

### 3.3 Válvula de Controle — Kv (IEC 60534)

$$\Delta P = \left(\frac{Q}{K_{v,ef}}\right)^2\frac{\rho}{1000}, \quad K_v \approx C_v \times 0{,}865$$

Múltiplas válvulas de controle em **série** somam suas perdas de pressão diretamente.

### 3.4 Curva da Bomba e Leis de Semelhança

O usuário insere 5 pontos Q×H na rotação nominal. Para outras frequências de inversor, aplica-se:

$$Q_2 = Q_1 \cdot \frac{n_2}{n_1}, \quad H_2 = H_1 \cdot \left(\frac{n_2}{n_1}\right)^2, \quad \frac{n_2}{n_1} = \frac{f_2}{f_1}$$

O ponto de operação é a interseção da curva da bomba com a curva do sistema. Três curvas do sistema são plotadas: abertura 20%, 100% e abertura definida pelo usuário.

---

## 4. Referências

- Incropera et al. — *Fundamentals of Heat and Mass Transfer*, 7ª ed.
- ISO 4006 / ABNT NBR 12213 — Darcy-Weisbach
- IEC 60534 — Coeficiente Kv
- API MPMS Chapter 4 — Calibração de medidores de óleo

---
*Validar com cálculos detalhados antes da especificação final.*
""")
    else:
        st.markdown(r"""
# 📖 User Manual and Calculation Memorandum

## 1. Overview

Parametric sizing tool for a closed-loop oil meter calibration laboratory
(1 tank, 1 centrifugal pump + VFD, bare piping, 0–800 m³/h).

| Tab | Function |
|-----|----------|
| 🌡️ Thermal Simulation | Heating and calibration window |
| 💧 System Curve | Head loss, hot & cold H×Q curves |
| 📋 BOM | Exportable bill of materials (CSV) |

---

## 2. Thermal Simulation

### 2.1 Pump Power — Assumption

Power is held **constant** (direct kW input). Centrifugal pump power varies little across the H×Q
curve. The engineer chooses max or nominal power.

### 2.2 Effect of Flow Rate on Temperature Curve

Flow rate directly affects the temperature curve:

$$Q \uparrow \;\Rightarrow\; Re \uparrow \;\Rightarrow\; Nu \uparrow \;\Rightarrow\; h_{int} \uparrow \;\Rightarrow\; R_{conv,int} \downarrow \;\Rightarrow\; Q_{loss} \uparrow \;\Rightarrow\; dT/dt \downarrow$$

**Higher flow → greater heat loss → slower heating.**
Phases with different flow rates will produce temperature curves with distinctly different slopes.

### 2.3 Thermal Balance — Explicit Euler (Δt = 1 s)

$$m \cdot c_p \cdot \frac{dT_f}{dt} = \dot{W}_p - Q_{loss}$$

- $\dot{W}_p = P_{pump} \cdot \eta \cdot f_{heat}$ [W] — constant per phase
- $m = V_{total} \cdot \rho$ [kg]

### 2.4 Thermal Resistances

**Internal convection (Dittus-Boelter):**
$$Nu = 0.023 \cdot Re^{0.8} \cdot Pr^{0.33}$$

**Wall conduction:** $R_{cond} = \ln(D_o/d_i)\,/\,(2\pi k_{steel} L)$, $k_{steel}=45$ W/m·K

**External convection:** $R_{ext} = 1\,/\,(h_{ext}\pi D_o L)$

### 2.5 Surface Temperature — Newton-Raphson

$$\frac{T_f - T_s}{R_{int}} = \frac{T_s - T_{amb}}{R_{ext}} + \varepsilon\sigma A(T_s^4 - T_{amb}^4)$$

Solved iteratively; heat loss: $Q_{loss} = (T_f - T_s)/R_{int}$

### 2.6 Calibration Condition

| Event | Criterion |
|-------|-----------|
| Calibration start | $\mu(T_{110\%}) = 1.10\,\mu_{nom}$ |
| Calibration end | $\mu(T_{90\%}) = 0.90\,\mu_{nom}$ |
| Window | $\Delta t = t_{90\%} - t_{110\%}$ |

---

## 3. System Curve

### 3.1 Darcy-Weisbach + Colebrook-White

$$h_f = f \cdot \frac{L}{d_i} \cdot \frac{V^2}{2g}; \quad \frac{1}{\sqrt{f}}=-2\log_{10}\!\left(\frac{\varepsilon_r}{3.7}+\frac{2.51}{Re\sqrt{f}}\right)$$

For $Re<2300$: $f=64/Re$.

### 3.2 Minor Losses (K method)

| Component | K |
|---|---|
| 90° elbow | 0.30 |
| 45° elbow | 0.20 |
| Tee | 0.50 |
| Ball valve (isolation) | 0.10 |
| Butterfly valve (isolation) | 0.50 |

### 3.3 Control Valve — Kv (IEC 60534)

$$\Delta P = \left(\frac{Q}{K_{v,eff}}\right)^2\frac{\rho}{1000}, \quad K_v \approx C_v \times 0.865$$

Multiple control valves in **series** sum their pressure drops directly.

### 3.4 Pump Curve and Affinity Laws

The user enters 5 Q×H points at nominal speed. For other VFD frequencies:

$$Q_2 = Q_1 \cdot \frac{n_2}{n_1}, \quad H_2 = H_1 \cdot \left(\frac{n_2}{n_1}\right)^2, \quad \frac{n_2}{n_1} = \frac{f_2}{f_1}$$

The operating point is the intersection of the pump curve and the system curve. Three system curves are plotted: 20% opening, 100% opening, and user-defined opening.

---

## 4. References

- Incropera et al. — *Fundamentals of Heat and Mass Transfer*, 7th ed.
- ISO 4006 / ABNT NBR 12213 — Darcy-Weisbach
- IEC 60534 — Kv flow coefficient
- API MPMS Chapter 4 — Oil meter calibration

---
*Always validate with detailed engineering calculations before final specification.*
""")
