ENGINE_ID_COLUMN = "engine_id"
CYCLE_COLUMN = "cycle"

NON_FEATURE_COLUMNS = {"RUL", "life_ratio", CYCLE_COLUMN, ENGINE_ID_COLUMN}

OPERATIONAL_SETTING_NAMES = ["setting_1", "setting_2", "setting_3"]

# Sensor names from the paper
SENSOR_NAMES = [
    "T2",  # Total temperature at fan inlet
    "T24",  # Total temperature at LPC outlet
    "T30",  # Total temperature at HPC outlet
    "T50",  # Total temperature at LPT outlet
    "P2",  # Pressure at fan inlet
    "P15",  # Total pressure in bypass duct
    "P30",  # Total pressure at HPC outlet
    "Nf",  # Physical fan speed
    "Nc",  # Physical core speed
    "epr",  # Engine pressure ratio
    "Ps30",  # Static pressure at HPC outlet
    "phi",  # Ratio of fuel flow to Ps30
    "NRf",  # Corrected fan speed
    "NRc",  # Corrected core speed
    "BPR",  # Bypass ratio
    "farB",  # Burner fuel-air ratio
    "htBleed",  # Bleed enthalpy
    "Nf_dmd",  # Demanded fan speed
    "PCNfR_dmd",  # Demanded corrected fan speed
    "W31",  # High-pressure turbine coolant bleed
    "W32",  # Low-pressure turbine coolant bleed
]

# Full raw CMAPSS file column layout
RAW_COLUMN_NAMES = [
    ENGINE_ID_COLUMN,
    CYCLE_COLUMN,
    *OPERATIONAL_SETTING_NAMES,
    *SENSOR_NAMES,
]
