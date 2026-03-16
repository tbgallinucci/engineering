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
import json
import os
from datetime import datetime

st.set_page_config(
    page_title="Calibration Lab Sizing Tool",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# SAVE / LOAD CONFIGURATIONS LOGIC
# ─────────────────────────────────────────────────────────────────────────────
CONFIG_FILE = "lab_configs.json"

def get_saved_configs():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_config(name):
    configs = get_saved_configs()
    data_to_save = {}
    for k, v in st.session_state.items():
        # Avoid saving internal dataframes, plot data, or buttons states
        if k not in ["th_data", "hy_data", "hy_sim_active"] and not k.endswith("btn"):
            if isinstance(v, (int, float, str, bool, list, dict)):
                data_to_save[k] = v
    configs[name] = data_to_save
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(configs, f, indent=4)

def load_config(name):
    configs = get_saved_configs()
    if name in configs:
        for k, v in configs[name].items():
            st.session_state[k] = v

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
        "r_eh": "⚡ Energia — Aquecimento",
        "r_ec": "⚡ Energia — Calibração",
        "r_et": "⚡ Energia Total",
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
        "kv_curve_help": "Insira 3 pontos da curva do fabricante: abertura (%) e Kv correspondente. A interpolação é log-linear, adequada para válvulas de equal-percentage.",
        "kv_op_j": "Abertura (%)",
        "kv_kv_j": "Kv (m³/h·bar⁰·⁵)",
        "n_ctrl_lbl": "Quantidade de válvulas de controle:",
        "n_ctrl_help": "Cada válvula de controle possui Kv e abertura independentes. Em loop fechado, válvulas em série somam as perdas de pressão.",
        "ctrl_series_note": "ℹ️ As válvulas de controle estão em **paralelo** no loop — os valores de Kv de cada linha são somados.",
        "ctrl_v": "Válvula de Controle",
        "op_lbl": "Abertura da válvula (%)",
        "op_help": "Deslize para ver o gráfico atualizar em tempo real.",
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
        "sys_base": "Base Pipe (Sem Válvula)",
        "pump_nom": "Bomba — {f} Hz (nominal)",
        "pump_fmin": "Bomba — {f} Hz (mín. inversor)",
        "pump_fmax": "Bomba — {f} Hz (máx. inversor)",
        "sim_hdr": "🎛️ Simulação Interativa (Atualização em Tempo Real)",
        "kv_calc": "**Calculadora rápida de Kv:**",
        "kv_Q": "Q (m³/h):",
        "kv_dP": "ΔP máx (bar):",
        "kv_rho": "ρ (kg/m³):",
        "kv_res": "**Kv mínimo: {kv:.0f} m³/h·bar⁰·⁵**",
        "hy_title": "Curva do Sistema – Altura Manométrica vs Vazão",
        "hy_px": "Vazão (m³/h)", "hy_py": "Altura Manométrica (m)",
        "seg_hdr": "Segmentos de Tubulação",
        "seg_note": "ℹ️ Cada segmento representa um trecho com diâmetro diferente. Vazão Q é conservada em série — apenas a velocidade muda. Rugosidade e desnível são globais (barra lateral).",
        "seg_add": "➕ Adicionar segmento",
        "seg_del": "🗑️ Remover último",
        "seg_dn":  "DN int. (m)",
        "seg_L":   "L (m)",
        "seg_c90": "Curvas 90°",
        "seg_tee": "Tês",
        "seg_ve":  "V.Esfera",
        "seg_red": "Reduções",
        "seg_dred":"D montante (m)",
        "seg_lbl": "Segmento",
        "seg_sum": "Σ comprimentos: {L:.1f} m | {n} segmento(s)",
        "cfg_hdr": "💾 Configurações",
        "cfg_save": "Salvar Atual",
        "cfg_load": "Carregar Salva",
        "cfg_name": "Nome da configuração:",
        "cfg_sel": "Selecione para carregar:",
        "cfg_succ": "Configuração salva com sucesso!",
        "cfg_lsucc": "Configuração carregada com sucesso!",
        "cfg_empty": "Nenhuma configuração salva.",
        "hide_ref": "👁️ Ocultar curvas de referência (20% e 100%)",
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
        "r_eh": "⚡ Energy — Heating",
        "r_ec": "⚡ Energy — Calibration",
        "r_et": "⚡ Total Energy",
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
        "kv_curve_help": "Enter 3 points from the manufacturer's curve: opening (%) and corresponding Kv. Log-linear interpolation is used, appropriate for equal-percentage valves.",
        "kv_op_j": "Opening (%)",
        "kv_kv_j": "Kv (m³/h·bar⁰·⁵)",
        "n_ctrl_lbl": "Number of control valves:",
        "n_ctrl_help": "Each control valve has independent Kv and opening. In a closed loop, series valves add their pressure drops directly.",
        "ctrl_series_note": "ℹ️ Control valves are in **parallel** in the loop — their effective Kv values are summed.",
        "ctrl_v": "Control Valve",
        "op_lbl": "Valve opening (%)",
        "op_help": "Slide to see the chart update in real-time.",
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
        "sys_base": "Base Pipe (No Valve)",
        "pump_nom": "Pump — {f} Hz (nominal)",
        "pump_fmin": "Pump — {f} Hz (VFD min)",
        "pump_fmax": "Pump — {f} Hz (VFD max)",
        "sim_hdr": "🎛️ Interactive Simulation (Real-time update)",
        "kv_calc": "**Quick Kv calculator:**",
        "kv_Q": "Q (m³/h):",
        "kv_dP": "Max ΔP (bar):",
        "kv_rho": "ρ (kg/m³):",
        "kv_res": "**Minimum Kv: {kv:.0f} m³/h·bar⁰·⁵**",
        "hy_title": "System Curve – Head vs Flow Rate",
        "hy_px": "Flow Rate (m³/h)", "hy_py": "Head (m)",
        "seg_hdr": "Piping Segments",
        "seg_note": "ℹ️ Each segment represents a pipe section with a different diameter. Flow Q is conserved in series — only velocity changes. Roughness and static head are global (sidebar).",
        "seg_add": "➕ Add segment",
        "seg_del": "🗑️ Remove last",
        "seg_dn":  "Inner diam. (m)",
        "seg_L":   "L (m)",
        "seg_c90": "90° elbows",
        "seg_tee": "Tees",
        "seg_ve":  "Ball valves",
        "seg_red": "Reductions",
        "seg_dred":"Upstream D (m)",
        "seg_lbl": "Segment",
        "seg_sum": "Σ lengths: {L:.1f} m | {n} segment(s)",
        "cfg_hdr": "💾 Configurations",
        "cfg_save": "Save Current",
        "cfg_load": "Load Saved",
        "cfg_name": "Configuration name:",
        "cfg_sel": "Select to load:",
        "cfg_succ": "Saved successfully!",
        "cfg_lsucc": "Loaded successfully!",
        "cfg_empty": "No saved configs.",
        "hide_ref": "👁️ Hide reference curves (20% and 100%)",
    },
}

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
    Re = max(Re, 1)
    if Re <= 2300:
        return 64 / Re
    f_turb = 0.25 / (math.log10(er/3.7 + 5.74/max(Re,1)**0.9))**2
    for _ in range(50):
        fn = (1/(-2*math.log10(er/3.7 + 2.51/(Re*math.sqrt(f_turb)))))**2
        if abs(fn - f_turb) < 1e-8: break
        f_turb = fn
    if Re >= 4000:
        return f_turb
    f_lam_2300  = 64 / 2300
    f_turb_4000 = 0.25 / (math.log10(er/3.7 + 5.74/4000**0.9))**2
    alpha = (Re - 2300) / (4000 - 2300)
    return f_lam_2300 + alpha * (f_turb_4000 - f_lam_2300)

def head_loss(Q, segments, dz, ctrl_valves, rho_f, mu_f, rug_global_mm):
    if Q <= 0:
        return 0, 0, 0, dz, 0

    rug = rug_global_mm / 1000.0
    Hd_total = 0.0
    Hl_total = 0.0

    for seg in segments:
        d   = seg['d']
        L   = seg['L']
        V   = (Q / 3600.0) / (math.pi * d**2 / 4.0)
        Re  = rho_f * V * d / mu_f
        f   = colebrook(Re, rug / d)
        Hd_total += f * (L / d) * V**2 / (2 * 9.81)
        K_red = 0.0
        if seg.get('red_n', 0) > 0 and seg.get('d_up', 0) > d:
            area_ratio = (d / seg['d_up']) ** 2
            K_red = 0.5 * (1.0 - area_ratio) * seg['red_n']
        elif seg.get('red_n', 0) > 0 and seg.get('d_up', 0) < d:
            area_ratio = (seg['d_up'] / d) ** 2
            K_red = (1.0 - area_ratio) ** 2 * seg['red_n']
        K_seg = (seg['c90'] * 0.3 + seg['tee'] * 0.5 +
                 seg['ve']  * 0.1 + K_red)
        Hl_total += K_seg * V**2 / (2 * 9.81)

    Hc = 0.0
    Kv_eq = 0.0
    tem_valvula = False
    for Kv, op in ctrl_valves:
        if Kv > 0:
            tem_valvula = True
            if op > 0:
                Kv_eq += Kv * (op / 100.0)
    if tem_valvula:
        Hc = (Q / Kv_eq)**2 * (rho_f / 1000.0) * 1e5 / (rho_f * 9.81) if Kv_eq > 0 else 9999.0

    return Hd_total + Hl_total + dz + Hc, Hd_total, Hl_total, dz, Hc

def hm(h):
    return f"{int(h)}h {int((h-int(h))*60)}min"


# ─────────────────────────────────────────────────────────────────────────────
# PDF REPORT BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def build_report_pdf(lang, th_data, hy_data, global_params):
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
    styles = getSampleStyleSheet()
    style_title   = ParagraphStyle('ReportTitle', fontSize=18, fontName='Helvetica-Bold', spaceAfter=6, alignment=TA_CENTER)
    style_subtitle= ParagraphStyle('Subtitle', fontSize=10, fontName='Helvetica', textColor=colors.HexColor('#555555'), spaceAfter=14, alignment=TA_CENTER)
    style_h1      = ParagraphStyle('H1', fontSize=13, fontName='Helvetica-Bold', spaceBefore=14, spaceAfter=6, textColor=colors.HexColor('#1a3a5c'))
    style_h2      = ParagraphStyle('H2', fontSize=11, fontName='Helvetica-Bold', spaceBefore=10, spaceAfter=4, textColor=colors.HexColor('#2c5f8a'))
    style_body    = ParagraphStyle('Body', fontSize=9, fontName='Helvetica', spaceAfter=4, leading=14)
    style_small   = ParagraphStyle('Small', fontSize=8, fontName='Helvetica', textColor=colors.HexColor('#666666'), spaceAfter=2)

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

    def tbl_noheader():
        return TableStyle([
            ('FONTNAME',   (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE',   (0,0), (-1,-1), 8),
            ('FONTNAME',   (0,0), (0,-1), 'Helvetica-Bold'),
            ('FONTNAME',   (2,0), (2,-1), 'Helvetica-Bold'),
            ('ROWBACKGROUNDS', (0,0), (-1,-1), [colors.white, colors.HexColor('#f0f4f8')]),
            ('GRID',       (0,0), (-1,-1), 0.3, colors.HexColor('#cccccc')),
            ('ALIGN',      (1,0), (-1,-1), 'CENTER'),
            ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 3),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    PLT_STYLE = {'figure.facecolor': 'white', 'axes.facecolor': 'white',
                 'axes.grid': True, 'grid.alpha': 0.35, 'axes.spines.top': False,
                 'axes.spines.right': False}

    def mpl_to_rl(fig_mpl, width_mm=170, height_mm=85):
        plt.rcParams.update(PLT_STYLE)
        buf2 = io.BytesIO()
        fig_mpl.savefig(buf2, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig_mpl)
        buf2.seek(0)
        return RLImage(buf2, width=width_mm*mm, height=height_mm*mm)

    def make_thermal_chart(d):
        plt.rcParams.update(PLT_STYLE)
        fig_m, ax = plt.subplots(figsize=(9, 4.2))
        ax.plot(d['t_hp'], d['T_hp'], color='orangered', lw=2, label='Aquecimento' if PT else 'Heating')
        ax.plot(d['tc_h'], d['Tf_c'], color='royalblue', lw=2, label='Calibração' if PT else 'Calibration')
        ax.axhline(d['T110'], color='purple', ls='--', lw=1, label=f"T 110%µ = {d['T110']:.1f}°C")
        if d['T90']:
            ax.axhline(d['T90'], color='green', ls='--', lw=1, label=f"T 90%µ = {d['T90']:.1f}°C")
        ax.axhline(d['T_eq'], color='orangered', ls=':', lw=1, label=f"T_eq = {d['T_eq']:.1f}°C")
        ax.axvline(d['t110_h'], color='purple', ls=':', lw=1)
        if d['t90_h']:
            ax.axvline(d['t90_h'], color='green', ls=':', lw=1)
            ax.axvspan(d['t110_h'], d['t90_h'], alpha=0.08, color='green', label='Janela calibração' if PT else 'Calib. window')
        ax.set_xlabel('Tempo (h)' if PT else 'Time (h)', fontsize=9)
        ax.set_ylabel('Temperatura (°C)' if PT else 'Temperature (°C)', fontsize=9)
        ax.set_title('Temperatura do Fluido vs Tempo' if PT else 'Fluid Temperature vs Time', fontsize=10, fontweight='bold')
        ax.legend(fontsize=7, ncol=3, loc='lower right')
        fig_m.tight_layout()
        return mpl_to_rl(fig_m)

    def make_hydraulic_chart(d):
        plt.rcParams.update(PLT_STYLE)
        fig_m, ax = plt.subplots(figsize=(9, 4.8))
        Qr = d['Qr']
        ax.plot(Qr, d.get('H_sys_base', []), color='gray', lw=1.5, ls='--', label='Base Pipe')
        
        # Only plot reference curves if they were rendered in the interactive UI
        if d.get('show_ref', True):
            ax.plot(Qr, d['H_sys20'],  color='steelblue', lw=2, label='Sistema 20%' if PT else 'System 20%')
            ax.plot(Qr, d['H_sys100'], color='royalblue', lw=2, label='Sistema 100%' if PT else 'System 100%')
        
        uop = d['user_op']
        if uop not in (20, 100) or not d.get('show_ref', True):
            ax.plot(Qr, d['H_sys_usr'], color='cornflowerblue', lw=1.5, ls=':', label=f"Sistema {uop}%" if PT else f"System {uop}%")
        
        ax.plot(d['Qmin'], d['Hmin'], color='#FFB300', lw=2, label=f"Bomba {d['pc_fmin']:.0f}Hz" if PT else f"Pump {d['pc_fmin']:.0f}Hz")
        ax.plot(d['Qmx'],  d['Hmx'],  color='#CC0000', lw=2, label=f"Bomba {d['pc_fmax']:.0f}Hz" if PT else f"Pump {d['pc_fmax']:.0f}Hz")
        if d['pc_freq0'] not in (d['pc_fmin'], d['pc_fmax']):
            ax.plot(d['Qnom'], d['Hnom'], color='orangered', lw=1.5, ls=':', label=f"Bomba {d['pc_freq0']:.0f}Hz" if PT else f"Pump {d['pc_freq0']:.0f}Hz")
        
        op_colors = {'20%': '#e67e00', f"{uop}%": '#8B008B', '100%': '#006400'}
        for freq_key, ops_t in [('fmin', d['ops_fmin_pts']), ('fnom', d['ops_fnom_pts']), ('fmax', d['ops_fmax_pts'])]:
            sym = {'fmin': 'v', 'fnom': 'o', 'fmax': '^'}[freq_key]
            items = [(f"{uop}%", ops_t[1])]
            if d.get('show_ref', True):
                items = [('20%', ops_t[0])] + items + [('100%', ops_t[2])]
                
            for lbl, (q_op, h_op) in items:
                if q_op is not None:
                    col = op_colors.get(lbl, 'black')
                    ax.plot(q_op, h_op, marker=sym, ms=8, color=col, zorder=5)
        if d['y_max']:
            ax.set_ylim(0, d['y_max'])
        ax.set_xlim(0, d['hy_qmax'] * 1.05)
        ax.set_xlabel('Vazão (m³/h)' if PT else 'Flow Rate (m³/h)', fontsize=9)
        ax.set_ylabel('Altura Manométrica (m)' if PT else 'Head (m)', fontsize=9)
        ax.set_title('Curva do Sistema e Bomba' if PT else 'System & Pump Curve', fontsize=10, fontweight='bold')
        ax.legend(fontsize=7, ncol=3, loc='upper right')
        fig_m.tight_layout()
        return mpl_to_rl(fig_m)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=20*mm, rightMargin=20*mm, topMargin=20*mm, bottomMargin=20*mm)
    story = []
    W = A4[0] - 40*mm

    story.append(Paragraph("🏭 Calibration Lab Sizing Tool", style_title))
    caption = ("Relatório de Dimensionamento — Laboratório de Calibração de Medidores de Óleo" if PT else "Sizing Report — Oil Meter Calibration Laboratory")
    story.append(Paragraph(caption, style_subtitle))
    story.append(Paragraph(f"{'Gerado em' if PT else 'Generated'}: {datetime.now().strftime('%d/%m/%Y %H:%M')}", style_small))
    story.append(HRFlowable(width='100%', thickness=1.5, color=colors.HexColor('#1a3a5c'), spaceAfter=10))

    gp = global_params
    story.append(Paragraph("1. " + ("Parâmetros Globais do Sistema" if PT else "Global System Parameters"), style_h1))
    gp_rows = [
        [("Fluido" if PT else "Fluid"), gp['fluid'], ("Diâmetro interno" if PT else "Inner diameter"), f"{gp['d_inner']:.4f} m"],
        [("Diâmetro externo" if PT else "Outer diameter"), f"{gp['D_outer']:.4f} m", ("Comprimento total" if PT else "Total length"), f"{gp['L_pipe']:.1f} m"],
        [("Rugosidade" if PT else "Roughness"), f"{gp['rug_mm']:.3f} mm", ("Desnível estático" if PT else "Static head"), f"{gp['dz_glob']:.1f} m"],
        [("Emissividade" if PT else "Emissivity"), f"{gp['eps_emit']:.2f}", ("Convecção externa" if PT else "External conv."), f"{gp['h_ext']:.1f} W/m²·K"],
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

    if hy_data:
        story.append(Paragraph("2. " + ("Curva do Sistema e Bomba" if PT else "System & Pump Curve"), style_h1))
        story.append(Paragraph("2.1 " + ("Parâmetros Hidráulicos" if PT else "Hydraulic Parameters"), style_h2))
        hy_inp = [
            [("Fluido — densidade" if PT else "Fluid — density"), f"{hy_data['hy_rho']:.0f} kg/m³", ("Viscosidade nominal" if PT else "Nominal viscosity"), f"{hy_data['hy_mu_cP']:.1f} cP"],
            [("Vazão máxima" if PT else "Max flow rate"), f"{hy_data['hy_qmax']:.0f} m³/h", ("Frequência nominal" if PT else "Nominal frequency"), f"{hy_data['pc_freq0']:.0f} Hz"],
            [("Freq. mínima inversor" if PT else "VFD min freq."), f"{hy_data['pc_fmin']:.0f} Hz", ("Rugosidade global" if PT else "Global roughness"), f"{hy_data.get('rug_mm', 0.046):.3f} mm"],
        ]
        th2 = Table(hy_inp, colWidths=[W*0.28, W*0.22, W*0.28, W*0.22])
        th2.setStyle(tbl_noheader())
        story.append(th2)
        story.append(Paragraph("2.2 " + ("Curva do Sistema e Pontos de Operação" if PT else "System Curve & Operating Points"), style_h2))
        story.append(make_hydraulic_chart(hy_data))
        story.append(Paragraph("2.3 " + ("Pontos de Operação — Rotação Variável" if PT else "Operating Points — Variable Speed"), style_h2))
        freq_labels = [
            f"{hy_data['pc_fmin']:.0f} Hz ({'mín.' if PT else 'min'})",
            f"{hy_data['pc_freq0']:.0f} Hz ({'nominal' if PT else 'rated'})",
            f"{hy_data['pc_fmax']:.0f} Hz ({'máx.' if PT else 'max'})",
        ]
        op_header = [("Abertura" if PT else "Opening"), ("Vazão" if PT else "Flow"), ("Altura" if PT else "Head")]
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
        wrap = Table([[op_tables[0][0], op_tables[1][0], op_tables[2][0]], [op_tables[0][1], op_tables[1][1], op_tables[2][1]]], colWidths=[W/3, W/3, W/3])
        wrap.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'TOP'), ('LEFTPADDING', (0,0), (-1,-1), 2), ('RIGHTPADDING', (0,0), (-1,-1), 2)]))
        story.append(wrap)

    story.append(PageBreak())
    if th_data:
        story.append(Paragraph("3. " + ("Simulação Térmica" if PT else "Thermal Simulation"), style_h1))
        story.append(Paragraph("3.1 " + ("Parâmetros de Entrada" if PT else "Input Parameters"), style_h2))
        inp_rows = [
            [("Volume total" if PT else "Total volume"), f"{th_data['vol_m3']:.1f} m³", ("Temp. ambiente" if PT else "Ambient temp."), f"{th_data['T_amb']:.1f} °C"],
            [("Viscosidade nominal" if PT else "Nominal viscosity"), f"{th_data['mu_nom_cP']:.1f} cP", ("Tempo simulação" if PT else "Simulation time"), f"{th_data['t_sim_h']:.1f} h"],
        ]
        ph_rows = [
            [("" if PT else ""), ("Fase Aquecimento" if PT else "Heating Phase"), ("Fase Calibração" if PT else "Calibration Phase")],
            [("Potência bomba" if PT else "Pump power"), f"{th_data['P_heat']:.1f} kW", f"{th_data['P_cal']:.1f} kW"],
            [("Vazão" if PT else "Flow rate"), f"{th_data['Q_heat']:.0f} m³/h", f"{th_data['Q_cal']:.0f} m³/h"],
            [("Eficiência" if PT else "Efficiency"), f"{th_data['ef_heat']:.0f}%", f"{th_data['ef_cal']:.0f}%"],
        ]
        ti = Table(inp_rows, colWidths=[W*0.25, W*0.25, W*0.25, W*0.25])
        ti.setStyle(tbl_noheader())
        story.append(ti)
        story.append(Spacer(1, 4))
        tp = Table(ph_rows, colWidths=[W*0.35, W*0.325, W*0.325])
        tp.setStyle(tbl_noheader())
        story.append(tp)
        story.append(Paragraph("3.2 " + ("Resultados" if PT else "Results"), style_h2))
        res_rows = [
            [("Parâmetro" if PT else "Parameter"), ("Valor" if PT else "Value")],
            [("Viscosidade alvo" if PT else "Target viscosity"),    f"{th_data['mu_nom_cP']:.1f} cP"],
            [("Temperatura alvo (100% µ)" if PT else "Target temp. (100% µ)"), f"{th_data['Tnom']:.1f} °C" if th_data['Tnom'] else "N/A"],
            [("T início calibração (110% µ)" if PT else "Calib. start temp (110% µ)"), f"{th_data['T110']:.1f} °C"],
            [("T fim calibração (90% µ)" if PT else "Calib. end temp (90% µ)"), f"{th_data['T90']:.1f} °C" if th_data['T90'] else "N/A"],
            [("Temperatura de equilíbrio" if PT else "Equilibrium temperature"), f"{th_data['T_eq']:.1f} °C"],
            [("Tempo de aquecimento" if PT else "Heating time"),    hm(th_data['t110_h'])],
            [("Janela de calibração" if PT else "Calibration window"), hm(th_data['cwin_h']) if th_data['cwin_h'] else ("Não atingida" if PT else "Not reached")],
            [("Energia — aquecimento" if PT else "Energy — heating"), f"{th_data['E_heat']:.1f} kWh"],
            [("Energia — calibração" if PT else "Energy — calibration"), f"{th_data['E_cal']:.1f} kWh" if th_data['E_cal'] is not None else ("Não atingida" if PT else "Not reached")],
            [("Energia total" if PT else "Total energy"), f"{th_data['E_total']:.1f} kWh" if th_data['E_total'] is not None else "—"],
        ]
        tr = Table(res_rows, colWidths=[W*0.6, W*0.4])
        tr.setStyle(tbl_style())
        story.append(tr)
        story.append(Paragraph("3.3 " + ("Curva de Temperatura" if PT else "Temperature Curve"), style_h2))
        story.append(make_thermal_chart(th_data))

    story.append(Spacer(1, 12))
    story.append(HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#aaaaaa'), spaceAfter=4))
    footer = ("*Validar com cálculos detalhados antes da especificação final.*" if PT else "*Always validate with detailed engineering calculations before final specification.*")
    story.append(Paragraph(footer, style_small))

    doc.build(story)
    buf.seek(0)
    return buf.read()


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

    use_manual = st.checkbox(S["manual_cb"], value=False, key="use_manual")
    if use_manual:
        rho      = st.number_input(S["density"], min_value=100.0, value=850.0, key="rho_man")
        cp_fluid = st.number_input(S["cp"],      min_value=0.1,   value=2000.0, key="cp_man")
        k_fluid  = st.number_input(S["k_lbl"],   min_value=0.01,  value=0.12, key="k_man")
        _mu      = st.number_input(S["mu_lbl"],  min_value=0.001, value=0.025, key="mu_man")
        viscosity_model = lambda Tf, m=_mu: m
        fluid_choice = "Manual"
    else:
        FLUIDS = [
            "KRD MAX 225 (11.4 - 40.8 cP)",
            "KRD MAX 2205 (82.5 - 402 cP)",
            "KRD MAX 685 (68.2 - 115.6 cP)",
            "KRD MAX 55 (2.4 - 4.64 cP)",
        ]
        fluid_choice = st.selectbox(S["sel_fluid"], FLUIDS, key="fluid_choice")
        rho = 850.0; cp_fluid = 2000.0; k_fluid = 0.12
        _vm = {
            FLUIDS[0]: lambda Tf: 0.1651  * np.exp(-0.046 * Tf),
            FLUIDS[1]: lambda Tf: 1.9133  * np.exp(-0.053 * Tf),
            FLUIDS[2]: lambda Tf: 0.5933  * np.exp(-0.054 * Tf),
            FLUIDS[3]: lambda Tf: -9e-08*Tf**3 + 1e-05*Tf**2 - 0.0007*Tf + 0.0165,
        }
        viscosity_model = _vm[fluid_choice]

    st.subheader(S["pipe_sub"])
    d_inner  = st.number_input(S["d_in"],    min_value=0.01, value=0.2571, key="d_inner")
    D_outer  = st.number_input(S["d_out"],   min_value=0.01, value=0.3238, key="D_outer")
    L_pipe   = st.number_input(S["L_p"],     min_value=1.0,  value=40.0, key="L_pipe")
    rug_mm   = st.number_input(S["hy_rug"],  min_value=0.001, value=0.046, help=S["hy_rug_h"], key="rug_mm")
    dz_glob  = st.number_input(S["hy_dz"],   value=2.0, key="dz_glob")
    eps_emit = st.number_input(S["eps_lbl"], min_value=0.01, max_value=1.0, value=0.85, help=S["eps_help"], key="eps_emit")
    h_ext    = st.number_input(S["hout_lbl"], min_value=1.0, value=10.0, help=S["hout_help"], key="h_ext")

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

    # ── SAVE/LOAD WIDGETS ──
    st.divider()
    st.subheader(S["cfg_hdr"])
    cfg_names = list(get_saved_configs().keys())

    with st.expander(S["cfg_save"]):
        new_name = st.text_input(S["cfg_name"], key="new_cfg_name")
        if st.button(S["cfg_save"], key="save_cfg_btn"):
            if new_name:
                save_config(new_name)
                st.success(S["cfg_succ"])
                st.rerun()

    with st.expander(S["cfg_load"]):
        if cfg_names:
            sel_cfg = st.selectbox(S["cfg_sel"], cfg_names, key="sel_cfg_name")
            if st.button(S["cfg_load"], key="load_cfg_btn"):
                load_config(sel_cfg)
                st.success(S["cfg_lsucc"])
                st.rerun()
        else:
            st.info(S["cfg_empty"])

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title(S["app_title"])
st.caption(S["app_caption"])

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_hy, tab_th, tab_mn = st.tabs([
    S["tab_hy"], S["tab_th"], S["tab_mn"]
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 – HYDRAULIC SYSTEM CURVE
# ─────────────────────────────────────────────────────────────────────────────
with tab_hy:
    st.header(S["hy_header"])

    # ── Segment table ──────────────────────────────────────────────────────────
    st.subheader(S["seg_hdr"])
    st.info(S["seg_note"])

    g1, g2, g3 = st.columns(3)
    g1.metric(S["hy_rug"], f"{rug_mm:.3f} mm")
    g2.metric(S["hy_dz"],  f"{dz_glob:.1f} m")
    g3.metric(S["d_out"],  f"{D_outer:.4f} m  (térmica)" if lang=="pt" else f"{D_outer:.4f} m  (thermal)")

    _SEG_DEFAULTS = [
        {'d': d_inner, 'L': L_pipe, 'c90': 4, 'tee': 2, 've': 3, 'red_n': 0, 'd_up': 0.30},
    ]
    if 'hy_segments' not in st.session_state:
        st.session_state['hy_segments'] = [s.copy() for s in _SEG_DEFAULTS]

    ba, bb = st.columns(2)
    if ba.button(S["seg_add"], key="seg_add_btn"):
        prev = st.session_state['hy_segments'][-1]
        st.session_state['hy_segments'].append(
            {'d': prev['d'], 'L': 10.0, 'c90': 0, 'tee': 0, 've': 0, 'red_n': 0, 'd_up': prev['d']*1.25})
        st.rerun()
    if bb.button(S["seg_del"], key="seg_del_btn", disabled=len(st.session_state['hy_segments']) <= 1):
        st.session_state['hy_segments'].pop()
        st.rerun()

    COL_W = [0.07, 0.12, 0.10, 0.10, 0.10, 0.10, 0.10, 0.14]
    hdr = st.columns(COL_W)
    for h, lbl in zip(hdr, [S["seg_lbl"], S["seg_dn"], S["seg_L"], S["seg_c90"], S["seg_tee"], S["seg_ve"], S["seg_red"], S["seg_dred"]]):
        h.markdown(f"**{lbl}**")

    for si, seg in enumerate(st.session_state['hy_segments']):
        cols = st.columns(COL_W)
        cols[0].markdown(f"**#{si+1}**")
        seg['d']     = cols[1].number_input("d",     value=seg['d'],     min_value=0.005, step=0.001, format="%.4f", label_visibility="collapsed", key=f"sd_{si}")
        seg['L']     = cols[2].number_input("L",     value=seg['L'],     min_value=0.0, label_visibility="collapsed", key=f"sL_{si}")
        seg['c90']   = cols[3].number_input("c90",   value=seg['c90'],   min_value=0, step=1, label_visibility="collapsed", key=f"sc90_{si}")
        seg['tee']   = cols[4].number_input("tee",   value=seg['tee'],   min_value=0, step=1, label_visibility="collapsed", key=f"stee_{si}")
        seg['ve']    = cols[5].number_input("ve",    value=seg['ve'],    min_value=0, step=1, label_visibility="collapsed", key=f"sve_{si}")
        seg['red_n'] = cols[6].number_input("red_n", value=seg['red_n'], min_value=0, step=1, label_visibility="collapsed", key=f"sredn_{si}")
        seg['d_up']  = cols[7].number_input("d_up",  value=seg['d_up'],  min_value=0.005, step=0.001, format="%.4f", label_visibility="collapsed", key=f"sdup_{si}")

    hy_segments = st.session_state['hy_segments']
    total_L = sum(s['L'] for s in hy_segments)
    st.caption(S["seg_sum"].format(L=total_L, n=len(hy_segments)))
    hy_dz = dz_glob

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

    n_ctrl = st.number_input(S["n_ctrl_lbl"], min_value=0, value=1, step=1, help=S["n_ctrl_help"], key="n_ctrl")

    fcv_curve_data = []

    for i in range(int(n_ctrl)):
        st.markdown(f"**{S['ctrl_v']} {i+1}**")
        st.caption(S["kv_curve_help"])
        default_cv_pts = [(25, 45), (50, 394), (100, 913)]
        cv_pts_op = []; cv_pts_kv = []
        cc = st.columns(3)
        for j, (dop, dkv) in enumerate(default_cv_pts):
            with cc[j]:
                st.markdown(f"**{S['pc_pt']} {j+1}**")
                op_j = st.number_input(S["kv_op_j"], value=float(dop), min_value=0.0, max_value=100.0, key=f"cvop_{i}_{j}")
                kv_j = st.number_input(S["kv_kv_j"], value=float(dkv), min_value=0.0, key=f"cvkv_{i}_{j}")
                cv_pts_op.append(op_j); cv_pts_kv.append(kv_j)

        import numpy as _np
        from scipy.interpolate import interp1d as _interp1d
        _ops = _np.array(cv_pts_op, dtype=float)
        _kvs = _np.array(cv_pts_kv, dtype=float)
        sort_i = _np.argsort(_ops)
        _ops, _kvs = _ops[sort_i], _kvs[sort_i]
        
        _kvs_safe = _np.where(_kvs <= 0, 1e-5, _kvs) # Previne log(0) se usuário inserir 0
        _log_kv = _np.log(_kvs_safe)
        
        _interp = _interp1d(_ops, _log_kv, kind='linear', fill_value=(_log_kv[0], _log_kv[-1]), bounds_error=False)
        fcv_curve_data.append((_ops, _kvs, _interp))

    st.subheader(S["fl_hy"])
    fc1, fc2, fc3 = st.columns(3)
    hy_rho    = fc1.number_input(S["hy_rho"],  min_value=100.0, value=850.0, key="hy_rho")
    hy_mu_cP  = fc2.number_input(S["hy_mu_h"], min_value=0.01, value=25.0, help=S["hy_mu_h_h"], key="hy_mu_cP")
    hy_mu_hot = hy_mu_cP / 1000.0
    hy_qmax   = fc3.number_input(S["hy_qmax"], min_value=50.0, value=900.0, help=S["hy_qmax_h"], key="hy_qmax")

    st.subheader(S["pump_curve_hdr"])
    st.caption(S["pump_curve_help"])

    pc1, pc2, pc3 = st.columns(3)
    pc_poles = pc1.selectbox(S["pc_poles"], [2, 4, 6, 8], index=1, help=S["pc_poles_help"], key="pc_poles")
    pc_freq0 = pc2.number_input(S["pc_freq"], min_value=10.0, value=60.0, help=S["pc_freq_help"], key="pc_freq0")
    pc_fmin  = pc3.number_input(S["pc_fmin"], min_value=5.0,  value=20.0, key="pc_fmin")
    pc_fmax  = pc3.number_input(S["pc_fmax"], min_value=10.0, value=60.0, key="pc_fmax")

    default_pts = [(0, 45), (200, 42), (400, 35), (600, 22), (800, 5)]
    pump_pts = []
    col_headers = st.columns(5)
    for i, (dq, dh) in enumerate(default_pts):
        with col_headers[i]:
            st.markdown(f"**{S['pc_pt']} {i+1}**")
            q_i = st.number_input(S["pc_Q_lbl"].format(i+1), value=float(dq), min_value=0.0, key=f"pcQ{i}")
            h_i = st.number_input(S["pc_H_lbl"].format(i+1), value=float(dh), min_value=0.0, key=f"pcH{i}")
            pump_pts.append((q_i, h_i))

    # =========================================================================
    # LÓGICA DE SIMULAÇÃO REATIVA (SEM BOTÃO)
    # =========================================================================
    st.markdown("---")
    st.subheader(S["sim_hdr"])
    
    # Hide Reference Curves Toggle
    hide_ref_curves = st.checkbox(S["hide_ref"], value=False, key="hide_ref_curves")
    show_ref = not hide_ref_curves

    # Slider interativo posicionado imediatamente acima do gráfico
    if n_ctrl > 0:
        user_op = st.slider(S["op_lbl"], 0, 100, 100, key="main_valve_slider", help=S["op_help"])
    else:
        user_op = 100

    from scipy.interpolate import CubicSpline
    from scipy.optimize import brentq
    import numpy as _np2

    Qr = np.linspace(0, hy_qmax, 400)

    # ── System curves at 3 valve openings + Base Pipe ─────────────────────
    def sys_curve_base(Q_arr):
        H = []
        for Q in Q_arr:
            h, *_ = head_loss(Q, hy_segments, hy_dz, [], hy_rho, hy_mu_hot, rug_mm)
            H.append(h)
        return _np2.array(H)

    def resolve_ctrl_valves(op_pct):
        cv_resolved = []
        for data in fcv_curve_data:
            _ops_c, _kvs_c, _interp_c = data
            kv_resolved = float(_np2.exp(_interp_c(op_pct)))
            cv_resolved.append((kv_resolved, 100)) # Kv já escalonado pela interpolação
        return cv_resolved

    def sys_curve(Q_arr, op_pct):
        cv_ov = resolve_ctrl_valves(op_pct)
        H = []
        for Q in Q_arr:
            h, *_ = head_loss(Q, hy_segments, hy_dz, cv_ov, hy_rho, hy_mu_hot, rug_mm)
            H.append(h)
        return _np2.array(H)

    H_sys_base = sys_curve_base(Qr)
    H_sys20    = sys_curve(Qr, 20)
    H_sys100   = sys_curve(Qr, 100)
    H_sys_usr  = sys_curve(Qr, user_op)

    # ── Pump curves via affinity laws ──────────────────────────────────────
    Qp = np.array([p[0] for p in pump_pts])
    Hp = np.array([p[1] for p in pump_pts])
    sort_idx = np.argsort(Qp)
    Qp, Hp = Qp[sort_idx], Hp[sort_idx]

    def pump_curve_at_freq(f_target):
        ratio = f_target / pc_freq0
        Q_sc = Qp * ratio
        H_sc = Hp * ratio**2
        cs = CubicSpline(Q_sc, H_sc, extrapolate=False)
        Q_full = np.linspace(0, Q_sc[-1], 400)
        H_full = cs(Q_full)
        H_full = np.where(H_full < 0, np.nan, H_full)
        return Q_full, H_full, cs, Q_sc[-1]

    Qnom, Hnom, cs_nom, Qmax_nom = pump_curve_at_freq(pc_freq0)
    Qmin, Hmin, cs_fmin, Qmax_min = pump_curve_at_freq(pc_fmin)
    Qmx,  Hmx,  cs_fmax, Qmax_mx  = pump_curve_at_freq(pc_fmax)

    def find_op(cs_pump_f, Qmax_f, sys_H_arr, sys_Q_arr=Qr):
        cs_sys = CubicSpline(sys_Q_arr, sys_H_arr, extrapolate=True)
        diff = lambda Q: float(cs_pump_f(Q)) - float(cs_sys(Q))
        Q_search = np.linspace(1, min(Qmax_f, hy_qmax), 300)
        d_vals = [diff(q) for q in Q_search]
        ops = []
        for j in range(len(d_vals)-1):
            if d_vals[j] * d_vals[j+1] < 0:
                try:
                    q_op = brentq(diff, Q_search[j], Q_search[j+1])
                    h_op = float(cs_sys(q_op))
                    ops.append((q_op, h_op))
                except Exception:
                    pass
        return ops[-1] if ops else (None, None)

    op20   = find_op(cs_nom, Qmax_nom, H_sys20)
    op100  = find_op(cs_nom, Qmax_nom, H_sys100)
    op_usr = find_op(cs_nom, Qmax_nom, H_sys_usr)

    # ── Plot ──────────────────────────────────────────────────────────────
    fh = go.Figure()

    fh.add_trace(go.Scatter(x=Qr, y=H_sys_base, mode='lines',
        name=S["sys_base"], line=dict(color='gray', width=2, dash='dash')))

    if show_ref:
        fh.add_trace(go.Scatter(x=Qr, y=H_sys20, mode='lines',
            name=S["sys20"], line=dict(color='steelblue', width=2.5)))
        fh.add_trace(go.Scatter(x=Qr, y=H_sys100, mode='lines',
            name=S["sys100"], line=dict(color='royalblue', width=2.5)))
    
    # Always plot the user curve unless it perfectly overlaps with 20/100 and references are shown
    if user_op not in (20, 100) or not show_ref:
        fh.add_trace(go.Scatter(x=Qr, y=H_sys_usr, mode='lines',
            name=S["sys_user"] + f" ({user_op}%)",
            line=dict(color='cornflowerblue', width=2.5 if not show_ref else 2, dash='solid' if not show_ref else 'dot')))

    fh.add_trace(go.Scatter(x=Qmin, y=Hmin, mode='lines',
        name=S["pump_fmin"].format(f=pc_fmin), line=dict(color='#FFB300', width=2.5)))
    fh.add_trace(go.Scatter(x=Qmx, y=Hmx, mode='lines',
        name=S["pump_fmax"].format(f=pc_fmax), line=dict(color='#CC0000', width=2.5)))
    if pc_freq0 not in (pc_fmin, pc_fmax):
        fh.add_trace(go.Scatter(x=Qnom, y=Hnom, mode='lines',
            name=S["pump_nom"].format(f=pc_freq0), line=dict(color='orangered', width=2, dash='dot')))

    def ops_for_freq(cs_f, Qmax_f):
        return (find_op(cs_f, Qmax_f, H_sys20), find_op(cs_f, Qmax_f, H_sys_usr), find_op(cs_f, Qmax_f, H_sys100))

    ops_fmin = ops_for_freq(cs_fmin, Qmax_min)
    ops_fnom = (op20, op_usr, op100)
    ops_fmax = ops_for_freq(cs_fmax, Qmax_mx)

    op_valve_colors = {'20%': '#e67e00', f'{user_op}%': '#8B008B', '100%': '#006400'}
    op_speed_symbols = {'fmin': ('triangle-down', pc_fmin), 'fnom': ('circle', pc_freq0), 'fmax': ('triangle-up', pc_fmax)}

    for speed_key, ops_tuple in [('fmin', ops_fmin), ('fnom', ops_fnom), ('fmax', ops_fmax)]:
        sym, freq_val = op_speed_symbols[speed_key]
        
        # Filter points to plot based on the toggle
        items_to_plot = [(f'{user_op}%', ops_tuple[1])]
        if show_ref:
            items_to_plot = [('20%', ops_tuple[0])] + items_to_plot + [('100%', ops_tuple[2])]
            
        for valve_lbl, (q_op, h_op) in items_to_plot:
            if q_op is not None:
                col = op_valve_colors.get(valve_lbl, 'black')
                fh.add_trace(go.Scatter(
                    x=[q_op], y=[h_op], mode='markers+text',
                    marker=dict(color=col, size=13, symbol=sym, line=dict(color='white', width=1.5)),
                    text=[f"  <b>{valve_lbl} @ {freq_val}Hz</b><br>  Q={q_op:.0f} m³/h | H={h_op:.1f} m"],
                    textfont=dict(color=col, size=11), textposition='middle right', showlegend=False
                ))

    # Calculate and draw dP Annotations only for visible nominal points
    cs_sys_base = CubicSpline(Qr, H_sys_base, extrapolate=True)
    items_to_annotate = [(f'{user_op}%', op_usr)]
    if show_ref:
        items_to_annotate = [('20%', op20)] + items_to_annotate + [('100%', op100)]
        
    for valve_lbl, (q_op, h_op) in items_to_annotate:
        if q_op is not None:
            h_base_op = float(cs_sys_base(q_op))
            dP_bar = (h_op - h_base_op) * hy_rho * 9.81 / 100000.0

            fh.add_shape(type="line",
                x0=q_op, y0=h_base_op, x1=q_op, y1=h_op,
                line=dict(color=op_valve_colors.get(valve_lbl, 'black'), width=2, dash="dot")
            )

            fh.add_annotation(
                x=q_op, y=(h_op + h_base_op)/2,
                text=f"ΔP={dP_bar:.1f} bar",
                showarrow=False,
                xanchor="left",
                xshift=8,
                font=dict(color=op_valve_colors.get(valve_lbl, 'black'), size=11, family="Arial")
            )

    fh.add_vline(x=hy_qmax, line_dash="dot", line_color="gray", annotation_text=f"Q={hy_qmax:.0f} m³/h", annotation_position="top right")

    all_pump_H = list(Hnom[~np.isnan(Hnom)]) + list(Hmin[~np.isnan(Hmin)]) + list(Hmx[~np.isnan(Hmx)])
    y_max = max(all_pump_H) * 1.15 if all_pump_H else None

    fh.update_layout(
        title=S["hy_title"], xaxis_title=S["hy_px"], yaxis_title=S["hy_py"],
        xaxis=dict(range=[0, hy_qmax * 1.05]), yaxis=dict(range=[0, y_max]),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        template="plotly_white", hovermode="x unified", height=560)
    
    st.plotly_chart(fh, use_container_width=True)

    if any(q is not None for q, _ in [op20, op100, op_usr]):
        st.subheader(S["op_pt_sys"])
        def make_op_df(ops_tuple):
            o20_, ousr_, o100_ = ops_tuple
            rows_ = []
            
            # Filter rows based on the toggle
            items = [(f"{user_op}%", ousr_)]
            if show_ref:
                items = [("20%", o20_)] + items + [("100%", o100_)]
                
            for lbl, (q_op, h_op) in items:
                rows_.append({
                    S["op_lbl"]: lbl, S["op_Q"]: f"{q_op:.1f} m³/h" if q_op is not None else "—",
                    S["op_H"]: f"{h_op:.1f} m" if h_op is not None else "—",
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

        def ops_as_list(ops_tuple):
            o20_, ousr_, o100_ = ops_tuple
            return [("20%", o20_), (f"{user_op}%", ousr_), ("100%", o100_)]

        st.session_state['hy_data'] = {
            'hy_rho': hy_rho, 'hy_mu_cP': hy_mu_cP, 'hy_qmax': hy_qmax,
            'pc_freq0': pc_freq0, 'pc_fmin': pc_fmin, 'pc_fmax': pc_fmax,
            'ops_fmin': ops_as_list(ops_fmin), 'ops_fnom': ops_as_list(ops_fnom), 'ops_fmax': ops_as_list(ops_fmax),
            'Qr': Qr, 'H_sys_base': H_sys_base, 'H_sys20': H_sys20, 'H_sys100': H_sys100, 'H_sys_usr': H_sys_usr,
            'user_op': user_op, 'Qnom': Qnom, 'Hnom': Hnom, 'Qmin': Qmin, 'Hmin': Hmin, 'Qmx': Qmx,  'Hmx': Hmx,
            'ops_fmin_pts': ops_fmin, 'ops_fnom_pts': ops_fnom, 'ops_fmax_pts': ops_fmax,
            'y_max': y_max, 'segments': [{k: v for k, v in s.items()} for s in hy_segments],
            'rug_mm': rug_mm, 'dz_glob': dz_glob, 'show_ref': show_ref
        }

# ─────────────────────────────────────────────────────────────────────────────
with tab_th:
    st.header(S["th_header"])
    st.subheader(S["sys_data"])
    s1, s2, s3, s4 = st.columns(4)
    vol_m3     = s1.number_input(S["vol"],     min_value=0.1, value=10.0, key="vol_m3")
    T_amb      = s2.number_input(S["tamb"],    value=25.0, key="T_amb")
    mu_nom_cP  = s3.number_input(S["mu_nom"], value=25.0, min_value=0.01, key="mu_nom_cP_th")
    t_sim_h    = s4.number_input(S["tsim"],    min_value=0.1, value=10.0, key="t_sim_h")

    st.divider()
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

    if st.button(S["run"], type="primary", key="btn_run_thermal"):
        mu_110_pa = mu_nom_cP * 1.1 / 1000.0
        mu_90_pa  = mu_nom_cP * 0.9 / 1000.0
        T110 = solve_visc_temp(viscosity_model, mu_110_pa)
        T90  = solve_visc_temp(viscosity_model, mu_90_pa)
        Tnom = solve_visc_temp(viscosity_model, mu_nom_cP / 1000.0)

        if T110 is None or T90 is None:
            st.error(S["err_mu"]); st.stop()

        W_heat  = P_heat  * (ef_heat/100) * hf_heat * 1000.0   
        F_heat  = Q_heat  / 3600.0                             
        m_fluid = vol_m3  * rho

        dt    = 1.0
        t_max = t_sim_h * 3600.0
        time  = np.arange(0, t_max, dt)
        Tf_h  = np.zeros(len(time)); Tf_h[0] = T_amb

        for i in range(1, len(time)):
            dTdt = euler_step(Tf_h[i-1], T_amb, F_heat, W_heat, rho, cp_fluid, k_fluid, d_inner, D_outer, L_pipe, eps_emit, h_ext, m_fluid)
            Tf_h[i] = Tf_h[i-1] + dTdt * dt

        idx110 = np.where(Tf_h >= T110)[0]
        if len(idx110) == 0:
            st.warning(S["warn_110"]); st.stop()

        t110_s  = time[idx110[0]]
        T110_v  = Tf_h[idx110[0]]
        t110_h  = t110_s / 3600.0
        mask_h  = time <= t110_s
        t_hp    = time[mask_h]; T_hp = Tf_h[mask_h]

        W_cal  = P_cal * (ef_cal/100) * hf_cal * 1000.0
        F_cal  = Q_cal / 3600.0

        tc     = np.arange(t110_s, t_max, dt)
        Tf_c   = np.zeros(len(tc)); Tf_c[0] = T110_v

        for i in range(1, len(tc)):
            dTdt = euler_step(Tf_c[i-1], T_amb, F_cal, W_cal, rho, cp_fluid, k_fluid, d_inner, D_outer, L_pipe, eps_emit, h_ext, m_fluid)
            Tf_c[i] = Tf_c[i-1] + dTdt * dt

        T_eq  = Tf_c[-1]
        idx90 = np.where(Tf_c >= T90)[0]
        if len(idx90) > 0:
            t90_s = tc[idx90[0]]; T90_v = Tf_c[idx90[0]]
            t90_h = t90_s / 3600.0; cwin_h = t90_h - t110_h
        else:
            t90_h = T90_v = cwin_h = None

        st.subheader(S["res_hdr"])
        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric(S["r_mu"],  f"{mu_nom_cP:.1f} cP")
        r2.metric(S["r_Tt"],  f"{Tnom:.1f} °C" if Tnom else "N/A")
        r3.metric(S["r_T90"], f"{T90:.1f} °C"  if T90  else "N/A")
        r4.metric(S["r_T110"],f"{T110:.1f} °C")
        r5.metric(S["r_Teq"], f"{T_eq:.1f} °C")

        E_heat  = P_heat * t110_h                                        
        E_cal   = P_cal  * cwin_h if cwin_h is not None else None        
        E_total = E_heat + E_cal  if E_cal  is not None else None

        t1, t2, t3, t4, t5 = st.columns(5)
        t1.metric(S["r_th"], hm(t110_h))
        t2.metric(S["r_cw"], hm(cwin_h) if cwin_h is not None else S["na"])
        t3.metric(S["r_eh"], f"{E_heat:.1f} kWh")
        t4.metric(S["r_ec"], f"{E_cal:.1f} kWh"   if E_cal   is not None else S["na"])
        t5.metric(S["r_et"], f"{E_total:.1f} kWh"  if E_total is not None else S["na"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t_hp/3600, y=T_hp, mode='lines', name=S["tr_heat"], line=dict(color='orangered', width=2.5)))
        fig.add_trace(go.Scatter(x=tc/3600, y=Tf_c, mode='lines', name=S["tr_calib"], line=dict(color='royalblue', width=2.5)))

        xf = [0, t_sim_h]
        fig.add_trace(go.Scatter(x=xf, y=[T_eq,T_eq], mode='lines', name=f'T_eq={T_eq:.1f}°C', line=dict(color='orangered', dash='dash')))
        fig.add_trace(go.Scatter(x=xf, y=[T110,T110], mode='lines', name=f'T 110%µ={T110:.1f}°C', line=dict(color='purple', dash='dot')))
        fig.add_trace(go.Scatter(x=[t110_h,t110_h], y=[T_amb-5, T_eq+10], mode='lines', name=f't₁₁₀={hm(t110_h)}', line=dict(color='purple', dash='dot')))
        fig.add_trace(go.Scatter(x=[t110_h], y=[T110_v], mode='markers', marker=dict(color='purple', size=9), showlegend=False))

        if T90 is not None:
            fig.add_trace(go.Scatter(x=xf, y=[T90,T90], mode='lines', name=f'T 90%µ={T90:.1f}°C', line=dict(color='green', dash='dot')))
        if t90_h is not None:
            fig.add_trace(go.Scatter(x=[t90_h,t90_h], y=[T_amb-5, T_eq+10], mode='lines', name=f't₉₀={hm(t90_h)}', line=dict(color='green', dash='dot')))
            fig.add_trace(go.Scatter(x=[t90_h], y=[T90_v], mode='markers', marker=dict(color='green', size=9), showlegend=False))
            fig.add_vrect(x0=t110_h, x1=t90_h, fillcolor="green", opacity=0.07, layer="below", line_width=0, annotation_text=S["cw_lbl"], annotation_position="top left")

        fig.update_layout(title=S["pl_title"], xaxis_title=S["pl_x"], yaxis_title=S["pl_y"], legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0), hovermode="x unified", template="plotly_white", height=520)
        st.plotly_chart(fig, use_container_width=True)

        st.session_state['th_data'] = {
            'vol_m3': vol_m3, 'T_amb': T_amb, 'mu_nom_cP': mu_nom_cP,
            't_sim_h': t_sim_h, 'P_heat': P_heat, 'Q_heat': Q_heat,
            'ef_heat': ef_heat, 'P_cal': P_cal, 'Q_cal': Q_cal, 'ef_cal': ef_cal,
            'Tnom': Tnom, 'T90': T90, 'T110': T110, 'T_eq': T_eq,
            't110_h': t110_h, 'cwin_h': cwin_h,
            'E_heat': E_heat, 'E_cal': E_cal, 'E_total': E_total,
            't_hp': t_hp / 3600.0, 'T_hp': T_hp,
            'tc_h': tc / 3600.0,   'Tf_c': Tf_c,
            't90_h': t90_h, 'T90_v': T90_v if t90_h is not None else None,
            'T110_v': T110_v,
        }

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
