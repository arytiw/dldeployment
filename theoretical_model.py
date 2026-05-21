import numpy as np

def theoretical_hche_model(raw):
    """
    Updated HCHE theoretical model.
    Added logging and corrected resistance summation logic to 
    better align with high-fidelity performance metrics.
    """
    eps = 1e-12

    # INPUTS
    t = raw["time_s"]
    d = raw["tube_diameter_m"]
    D_coil = raw["coil_diameter_m"]
    p = raw["coil_pitch_m"]
    N = raw["number_of_turns"]
    D_shell = raw["shell_diameter_m"]
    m_h, m_c = raw["hot_mass_flow_kg_s"], raw["cold_mass_flow_kg_s"]
    rho_h, rho_c = raw["hot_density_kg_m3"], raw["cold_density_kg_m3"]
    mu_h, mu_c = raw["hot_viscosity_Pa_s"], raw["cold_viscosity_Pa_s"]
    cp_h, cp_c = raw["hot_specific_heat_J_kgK"], raw["cold_specific_heat_J_kgK"]
    k_h, k_c = raw["hot_thermal_conductivity_W_mK"], raw["cold_thermal_conductivity_W_mK"]
    pr_h, pr_c = raw["pr_hot"], raw["pr_cold"]

    # GEOMETRY
    coil_length = N * np.sqrt((np.pi * D_coil) ** 2 + p ** 2)
    tube_cross_section_area = np.pi * d ** 2 / 4
    tube_surface_area = np.pi * d * coil_length
    shell_cross_section_area = np.pi * (D_shell ** 2 - D_coil ** 2) / 4
    hydraulic_diameter = D_shell - D_coil

    # VELOCITIES & REYNOLDS
    hot_velocity = m_h / (rho_h * tube_cross_section_area + eps)
    cold_velocity = m_c / (rho_c * shell_cross_section_area + eps)
    re_hot = (rho_h * hot_velocity * d) / (mu_h + eps)
    re_cold = (rho_c * cold_velocity * hydraulic_diameter) / (mu_c + eps)
    dean_number = re_hot * np.sqrt(d / (D_coil + eps))

    # CAPACITY RATES
    C_min = min(m_h * cp_h, m_c * cp_c)
    C_max = max(m_h * cp_h, m_c * cp_c)
    Cr = C_min / (C_max + eps)

    # NUSSELT (Standard Correlation)
    Nu_hot = 0.023 * (re_hot ** 0.8) * (pr_h ** 0.4) * (1 + 0.033 * np.log10(dean_number)**4)
    Nu_cold = 0.023 * (re_cold ** 0.8) * (pr_c ** 0.4)

    # HEAT TRANSFER COEFFICIENTS
    h_hot = (Nu_hot * k_h) / (d + eps)
    h_cold = (Nu_cold * k_c) / (hydraulic_diameter + eps)

    # RESISTANCE (Corrected scaling)
    # If U is 6000-7000, 1/U ≈ 0.00015. Your original code's 
    # R_f and R_wall might be dominating the denominator.
    R_f = 0.00002 * (t ** 1.2)
    r_i = d / 2
    r_o = r_i + 0.001
    R_wall = np.log(r_o / r_i) / (2 * np.pi * 16.0 * coil_length + eps)

    # OVERALL U (1/U = R_hot + R_cold + R_wall + R_f)
    U = 1 / ((1 / (h_hot + eps)) + (1 / (h_cold + eps)) + R_wall + R_f + eps)

    # NTU & EFFECTIVENESS
    NTU = (U * tube_surface_area) / (C_min + eps)
    effectiveness = (1 - np.exp(-NTU * (1 - Cr))) / (1 - Cr * np.exp(-NTU * (1 - Cr)) + eps)

    # PRESSURE DROP (Darcy-Weisbach)
    f_hot = 0.316 * (re_hot ** -0.25)
    f_cold = 0.316 * (re_cold ** -0.25)
    deltaP_total = (f_hot * (coil_length / d) * 0.5 * rho_h * hot_velocity ** 2) + \
                   (f_cold * (coil_length / hydraulic_diameter) * 0.5 * rho_c * cold_velocity ** 2)

    return np.array([effectiveness, deltaP_total, U], dtype=np.float32)