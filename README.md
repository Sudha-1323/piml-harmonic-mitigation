# Physics-Informed Harmonic Mitigation in Residential Power Systems

This repository contains the dataset and MATLAB code developed for the research work on predicting and reducing harmonic distortion in residential electrical systems.

## Overview
This work proposes a combined framework for:
- Predicting Total Harmonic Distortion (THD)
- Reducing harmonics using adaptive filter control

The approach integrates:
- AO-NFRS for feature relevance analysis  
- Physics-Informed Machine Learning (PIML) for THD prediction  
- Bayesian-Optimized ANFIS for filter switching  

## Dataset
The dataset (`newinput_1000.xlsx`) contains:
- Electrical measurements (current, voltage, power, etc.)
- Harmonic components (up to 11th order)
- THD values
- Temperature and humidity (microclimatic factors)

## Code Files
- `AONRFS.m` → Feature relevance analysis using AO-NFRS  
- `PIML_THD.m` → THD prediction using physics-informed learning  
- `BOANFIS.m` → ANFIS model with Bayesian optimization  
- `RealTime_FilterSwitch_ANFIS.m` → Real-time filter switching logic  

## How to Use
1. Load the dataset from the Excel file  
2. Run AO-NFRS preprocessing  
3. Train the PIML model for THD prediction  
4. Execute ANFIS controller for adaptive filter switching  

## Output
- Predicted THD values  
- Filter switching decisions (OFF / APF / HYBRID)  

## Notes
- This implementation is for research and academic purposes  
- The framework is designed for residential power system applications  
