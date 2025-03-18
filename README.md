# A multi-scale model of _Dunaliella salina_

## Introduction

This repository contains the code and data used to develop, evaluate and analyse a multi-scale model of _Dunaliella salina_, based on the *i*EC1700 and a dynamic model.

## Data

The data used in this study is available in the `data` directory. The data is organized as follows:

- experimental: Experimental data for the green algae, including biomass composition, growth media, and growth profiles from experimental procedures.
- models: Genome-scale metabolic model of _Dunaliella salina_ (both the original model and an adaptation where the biomass is dividided into active and storage biomass).
- parameters: Initial parameters used in the dynamic model, based on literature and empirical evidence.

## Code

The code used in this study is available in the `code` directory. The code is organized as follows:

- 'experimental_data_analysis.ipynb': Jupyter notebook used to evaluate the experimental data (https://doi.org/10.1039/D4FB00229F).
- 'create_dfba_model.py': Python script used generate a model with active and storage biomass, adapting the stoichiometric coefficients of biomass compounds.
- 'main': Python script to run simulations with the DFBA package. It can be used to run all trials (run_all_parallel), run just one condition (run_condition), and perform parameter optimization (optimize).
- 'rhs': Python script used to calculate the right-hand side of the dynamic model, i.e., it determines the lower and/or upper bounds of exchange reactions considered in the model.
- drhs: Python script used to get the dynamic equations for each state variable
- 'results_analysis.py': Python script to generate the plots under standard and optimized conditions.
- 'senstivity_analysis.py': Python script to perform sensitivity analysis of the model.
- 'condition_optimization.py': Python script to optimize the initial nitrate and phosphate concentrations.


To reproduce the results, the packages listed in `requirements.txt` are required.

## Results

The results of this study are available in the `results` directory. The results are organized as follows:

- 'biomass_concentrations': Plots with biomass concentration vs time for each trial.
- 'concentrations': CSV files with the concentrations of each compound in the model for each trial.
- 'experimental_analysis': boxplots showing the relation between experimental conditions and compound production.
- 'figures': Figures generated in the study.
- 'logs': Log files generated during the simulations.
- 'macros': Bar plots showing the carbohydrate, protein, and lipid profile in each trial, and respective residuals and scatter analysis.
- 'parameters': JSON and TXT files containing the optimized parameters.
- 'pigments': Bar plots showing the pigment profile in each trial, and respective residuals and scatter analysis.
- 'quotas': Nitrogen and Phosphorus quotas over time for each trial.
- 'trajectories': CSV file listing all trajectories (i.e., reaction's fluxes) for each trial.
- 'sensitivity': Sensitivity analysis results.

