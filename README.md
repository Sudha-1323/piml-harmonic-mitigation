# Physics-Informed Harmonic Mitigation in Residential Power Systems

This repository contains the dataset and MATLAB implementation developed for the research work on predicting and mitigating harmonic distortion in residential electrical systems using Physics-Informed Machine Learning and adaptive neuro-fuzzy control.

---

## Overview

This work proposes an integrated framework for:

- Predicting Total Harmonic Distortion (THD)
- Reducing harmonics using adaptive filter control

The framework combines:

- Auto-Optimized Neighbourhood Fuzzy Rough Sets (AO-NFRS)
- Physics-Informed Machine Learning (PIML)
- Bayesian-Optimized ANFIS (BO-ANFIS)

The proposed system improves harmonic prediction accuracy and enables stable real-time filter switching in residential power networks.

---

## Dataset Information

The dataset (`newinput_1000.xlsx`) contains 1000 measurements collected from common residential appliances under different operating conditions.

### Appliances Included

- Heater
- Fan
- LED Lamp
- Refrigerator
- Induction Stove
- Laptop

### Recorded Parameters

The dataset includes:

- Voltage
- RMS Current
- Active Power
- Reactive Power
- Power Factor
- Harmonic Magnitudes (1st–11th order)
- Harmonic Phase Angles
- THD Current
- THD Voltage
- Temperature
- Humidity
- Load Type Information

Environmental parameters were included to analyse the influence of microclimatic conditions on harmonic behaviour.

---

## Repository Structure

| File | Description |
|------|-------------|
| `newinput_1000.xlsx` | Measurement dataset |
| `AONRFS.m` | AO-NFRS feature relevance analysis |
| `PIML_THD.m` | Physics-informed THD prediction model |
| `BOANFIS.m` | Bayesian-optimized ANFIS training |
| `RealTime_FilterSwitch_ANFIS.m` | Real-time adaptive filter switching |
| `README.md` | Project documentation |

---
## Code Information

### AONRFS.m

Implements the Auto-Optimized Neighbourhood Fuzzy Rough Set (AO-NFRS) algorithm used to evaluate feature relevance and rank electrical and environmental variables influencing harmonic distortion.

### PIML_THD.m

Implements the Physics-Informed Machine Learning (PIML) model for prediction of Total Harmonic Distortion (THD). The model integrates measured harmonic data with physics-based constraints derived from the THD formulation.

### BOANFIS.m

Implements Bayesian-Optimized Adaptive Neuro-Fuzzy Inference System (BO-ANFIS) training for generation of adaptive harmonic mitigation rules and filter control decisions.

### RealTime_FilterSwitch_ANFIS.m

Executes the adaptive filter switching framework and generates real-time decisions among OFF, APF, and HYBRID filtering modes based on predicted THD levels.

## Methodology

The proposed framework follows these stages:

1. Data acquisition from residential appliances
2. AO-NFRS preprocessing and feature relevance analysis
3. Physics-informed machine learning for THD prediction
4. Bayesian optimization of ANFIS parameters
5. Adaptive filter switching based on predicted THD levels
6. Real-time harmonic mitigation analysis

---

## Requirements

The implementation was developed using:

- MATLAB R2023a or later

Required MATLAB Toolboxes:

- Deep Learning Toolbox
- Fuzzy Logic Toolbox
- Statistics and Machine Learning Toolbox

### Additional Dependencies

The implementation requires the following MATLAB toolboxes:

* Deep Learning Toolbox
* Fuzzy Logic Toolbox
* Statistics and Machine Learning Toolbox
* Optimization Toolbox (for Bayesian optimization)

No external Python packages are required.

---

## Usage Instructions

### Step 1: Load Dataset

Open MATLAB and load:

```matlab
newinput_1000.xlsx
```

### Step 2: Run AO-NFRS Preprocessing

Execute:

```matlab
AONRFS.m
```

### Step 3: Train Physics-Informed THD Prediction Model

Run:

```matlab
PIML_THD.m
```

### Step 4: Execute BO-ANFIS Controller

Run:

```matlab
BOANFIS.m
```

### Step 5: Perform Real-Time Filter Switching

Execute:

```matlab
RealTime_FilterSwitch_ANFIS.m
```

---

## Output

The framework produces:

- Predicted THD values
- Harmonic mitigation analysis
- Adaptive filter switching decisions:
  - OFF
  - APF
  - HYBRID
- Real-time switching visualization

---
## Reproducibility

To reproduce the results reported in the manuscript:

1. Load the dataset `newinput_1000.xlsx`.
2. Execute `AONRFS.m` for feature relevance analysis.
3. Run `PIML_THD.m` to train and evaluate the THD prediction model.
4. Execute `BOANFIS.m` to generate adaptive control rules.
5. Run `RealTime_FilterSwitch_ANFIS.m` to perform filter switching simulation and visualization.
6. Compare the generated outputs with the figures and tables reported in the manuscript.

## Citation

If you use this dataset or code in your research, please cite:

Sudha K, Muthumeenakshi K, Dhanasekaran S.  
*Physics-Informed Machine Learning and Adaptive Neuro-Fuzzy Control for Harmonic Mitigation in Residential Power Systems.*

---

## License

This repository is provided for academic and research purposes only.

---

## Notes

- This implementation is intended for research and educational use.
- The framework is designed for residential power system applications.
- The repository supports reproducibility of the results presented in the associated manuscript.
