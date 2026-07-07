# CMAPSS FD001 sensor reference

Reference notes for the NASA C-MAPSS turbofan degradation dataset, subset
FD001, as used by this project's remaining-useful-life model. Sensor codes
follow the original damage-propagation-modeling paper (Saxena et al., PHM
2008), simulating a 90k-lb-thrust-class twin-spool turbofan.

## Dataset context

- FD001 simulates 100 training engines at a single operating condition (sea
  level) with a single fault mode: high-pressure compressor (HPC)
  degradation. Each engine starts with unknown initial wear, runs to
  failure, and one row is one flight cycle.
- Because there is only one operating condition in FD001, the three
  operational settings (`setting_1`, `setting_2`, `setting_3`) barely vary
  and carry no useful signal here.
- The model predicts `life_ratio` = remaining useful life divided by the
  engine's total life, so 1.0 is a fresh engine and 0.0 is failure. The
  serving system flags predicted failure below 0.1.

## Sensor glossary

Temperatures are total temperatures in degrees Rankine (°R), pressures in
psia, and shaft speeds in rpm unless noted.

| Code | Meaning | Notes |
| --- | --- | --- |
| T2 | Total temperature at fan inlet | Constant in FD001 (single operating condition); no diagnostic value here. |
| T24 | Total temperature at LPC outlet | Rises as compressor efficiency is lost; a useful degradation indicator. |
| T30 | Total temperature at HPC outlet | Directly downstream of the degrading component in FD001. |
| T50 | Total temperature at LPT outlet | Exhaust-side temperature; trends with overall efficiency loss. |
| P2 | Pressure at fan inlet | Constant in FD001. |
| P15 | Total pressure in bypass duct | Nearly constant in FD001. |
| P30 | Total pressure at HPC outlet | Compressor discharge pressure. |
| Nf | Physical fan speed | Low-spool shaft speed. |
| Nc | Physical core speed | High-spool shaft speed; the spool driving the degrading HPC. |
| epr | Engine pressure ratio (P50/P2) | Constant in FD001. |
| Ps30 | Static pressure at HPC outlet | In this project's trained model, the most important sensor feature by SHAP. A running engine always maintains substantial static pressure at the compressor exit -- a reading at or near zero is physically impossible while the engine operates and indicates a failed sensor or data feed, not engine state. |
| phi | Ratio of fuel flow to Ps30 (pps/psia) | Fuel metering relative to compressor discharge pressure. |
| NRf | Corrected fan speed | Fan speed normalized to inlet conditions. |
| NRc | Corrected core speed | Core speed normalized to inlet conditions. |
| BPR | Bypass ratio | Bypass to core airflow ratio. |
| farB | Burner fuel-air ratio | Constant in FD001. |
| htBleed | Bleed enthalpy | Customer/service bleed extraction. |
| Nf_dmd | Demanded fan speed | Controller demand; constant in FD001. |
| PCNfR_dmd | Demanded corrected fan speed | Controller demand; constant in FD001. |
| W31 | HPT coolant bleed (lbm/s) | Turbine cooling flow. |
| W32 | LPT coolant bleed (lbm/s) | Turbine cooling flow. |

## Physical plausibility

- Temperatures (T24, T30, T50), pressures (P30, Ps30), and shaft speeds
  (Nf, Nc, NRf, NRc) of an operating engine are always far above zero. Any
  such sensor reporting 0 or a physically absurd value means instrumentation
  or data-pipeline failure, and every model output consuming that reading is
  suspect until the feed is fixed.
- Healthy operating ranges observed in the FD001 training data (per-sensor
  mean, standard deviation, min, max) are stored in
  `configs/agent/training_statistics.json`; the drift tool's z-scores are
  computed against exactly those statistics. FD001 sensors move within
  narrow bands -- |z| beyond about 4 essentially never occurs in healthy
  training data.

## How degradation shows up

- HPC degradation is a slow drift, not a step change: sensor values move
  gradually and monotonically for most of an engine's life, with the shift
  accelerating in the final tens of cycles before failure.
- A genuine degradation signal appears as a *coordinated, sustained* trend
  across several related sensors (e.g. rising outlet temperatures together
  with shifting compressor-discharge conditions). An abrupt jump confined to
  a single sensor -- while every related sensor stays normal -- is the
  signature of a sensor fault rather than engine wear.
- This project's model relies most heavily on the engine's cycle count at
  the start and end of life, while sensor evidence (led by Ps30) carries the
  most weight in mid life. A prediction made in a cycle-dominated phase can
  stay stable even when a sensor misbehaves, so prediction stability alone
  never proves input health.
