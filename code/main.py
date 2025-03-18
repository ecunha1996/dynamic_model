import argparse
import copy
import json
import logging
import math
import os
import random
import shutil
import sys
from collections import OrderedDict
from functools import partial
from os.path import join

import pandas as pd
import sympy as sp
from dfba import DfbaModel, ExchangeFlux, KineticVariable
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from pandas import DataFrame
from parallelbar import progress_imap, progress_map
from sklearn.metrics import r2_score
from sympy import Max, Abs, Min
from timeout_decorator import timeout
from tqdm import tqdm

os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.path.insert(0, "/home/src/")
from gsmmutils.experimental.exp_matrix import ExpMatrix
from gsmmutils.dynamic.initial_conditions import get_initial_conditions
from gsmmutils.model.COBRAmodel import MyModel
from rhs import get_bounds
from drhs import get_dynamic_expressions
from gsmmutils.dynamic.exchange_fluxes import get_exchange_fluxes
from gsmmutils.graphics.plot import plot_concentrations, generate_plot_for_data
# import warnings
# logging.getLogger('pandas').setLevel(logging.CRITICAL)
# warnings.filterwarnings("ignore")
# logging.getLogger('dfba').setLevel(logging.CRITICAL)
# logging.getLogger('jit').setLevel(logging.CRITICAL)
# logging.getLogger("setuptools").setLevel(logging.CRITICAL)
# logging.getLogger("distutils").setLevel(logging.CRITICAL)
# logging.basicConfig(level=logging.CRITICAL)
# warnings.filterwarnings("ignore", module="setuptools.*")


DATA_PATH = "../data"
RESULTS_PATH = f"../results"
matrix = ExpMatrix(f"{DATA_PATH}/experimental/Matriz- DCCR Dunaliella salina_dfba_new.xlsx", conditions="Resume")
matrix.matrix["OC"] = matrix.matrix["TC"]
del matrix.matrix["TC"]
matrix.conditions.index = [e.replace("TC", "OC") for e in matrix.conditions.index]



def select_random_conditions(conditions, num_to_select, mandatory_strings):
    """
    Selects a specified number of strings from a list, making some of them mandatory.

    Args:
        conditions (list): List of strings to select from.
        num_to_select (int): Number of strings to randomly select.
        mandatory_strings (list): List of strings that must be included in the selected strings.

    Returns:
        list: List of selected strings.
    """
    if len(mandatory_strings) > num_to_select:
        raise ValueError("Number of mandatory strings cannot exceed number of strings to select.")
    to_ignore = ["fachet", "Xi", "Yimei"]
    conditions = [e for e in conditions if not any(e.startswith(ignore) for ignore in to_ignore)]
    random.shuffle(conditions)
    selected = mandatory_strings[:]
    remaining = num_to_select - len(mandatory_strings)
    selected.extend(random.sample(conditions, remaining))
    return selected


def read_model() -> MyModel:
    """
    Reads the model from the xml file and sets the objective function to the biomass reaction, demands, and minimizes the nitrate and phosphorus uptake.
    Returns
        MyModel: The model with the objective function set to the biomass reaction and the nitrate and phosphorus uptake minimized.
    -------

    """
    stoichiometric_model = MyModel(join(DATA_PATH, "models/model_dfba_no_caro.xml"), "e_ActiveBiomass__cytop")
    stoichiometric_model.exchanges.EX_C00011__dra.bounds = (-10000, 10000)
    stoichiometric_model.solver = "glpk"
    [setattr(x, 'objective_coefficient', 0) for x in stoichiometric_model.reactions if x.objective_coefficient != 0]

    objectives = {
        "e_ActiveBiomass__cytop": 1,
        "DM_C00369__chlo": 1,
        "DM_C05306__chlo": 10, "DM_C05307__chlo": 10,
        "DM_C08601__chlo": 1, "DM_C02094__chlo": 1,
        # "DM_C00116__cytop": 1,
        "DM_C00422__lip": 1,
         "EX_C00244__dra": -1, "EX_C00009__dra": -1,
        # "DM_C00244__cytop": 1
    }
    for reaction_id, value in objectives.items():
        stoichiometric_model.reactions.get_by_id(reaction_id).objective_coefficient = value




    return stoichiometric_model


fba_model = read_model()


def get_kinetic_variables() -> dict:
    """
    Creates the kinetic variables used in the model.
    Returns
    -------
    dict: Dictionary of kinetic variables.
    """
    light = KineticVariable("Light")
    X = KineticVariable("Biomass")
    F = KineticVariable("ActiveBiomass")
    nitrate = KineticVariable("Nitrate")
    n_quota = KineticVariable("Nitrogen_quota")
    phosphorus = KineticVariable("Phosphate")
    p_quota = KineticVariable("Phosphate_quota")
    starch = KineticVariable("Starch")
    starch_concentration = KineticVariable("Starch_concentration")
    tag = KineticVariable("TAG")
    # glycerol = KineticVariable("Glycerol")
    carotene = KineticVariable("Carotene")
    lutein = KineticVariable("Lutein")
    chlorophyll = KineticVariable("Chlorophyll")
    return {"X": X, "F": F, "nitrate": nitrate, "phosphorus": phosphorus,
            "starch": starch,
            "starch_concentration": starch_concentration,
            "chlorophyll": chlorophyll,
            "carotene": carotene,
            "n_quota": n_quota, "p_quota": p_quota,
            "tag": tag,
            # "glycerol": glycerol,
            "light": light, "lutein": lutein}


def generate_all_plots(condition: str, concentrations: DataFrame, experimental: list, trajectories: DataFrame):
    """
    Generates all the plots for a given condition.
    Parameters
    ----------
    condition: str
        Condition to generate the plots for.
    concentrations: DataFrame
        A DataFrame with concentrations.
    experimental: list
        A List with the  experimental data.
    trajectories: DataFrame
        A DataFrame with the trajectories.

    Returns
    -------

    """
    if  'Caro' in matrix.matrix[condition].columns:
        if condition.startswith("fachet") or condition.startswith("Xi") or condition.startswith("Yimei"):
            molecules = ['Caro']
            experimental_caro = [[matrix.matrix[condition][molecule].dropna().index.astype(float).tolist(),
                                  matrix.matrix[condition][molecule].dropna().tolist()] for molecule in molecules]
            plot_concentrations(concentrations, y=['Carotene'], experimental=experimental_caro,
                                filename=f"{RESULTS_PATH}/pigments/carotene_{condition}.png",
                                y_label="Macromolecule (g/gDW)", experimental_label=molecules)

    if 'Chl' in matrix.matrix[condition].columns:
        if condition.startswith("fachet") or condition.startswith("Xi") or condition.startswith("Yimei"):
            molecules = ['Chl']
            experimental_caro = [[matrix.matrix[condition][molecule].dropna().index.astype(float).tolist(),
                                  matrix.matrix[condition][molecule].dropna().tolist()] for molecule in molecules]
            plot_concentrations(concentrations, y=['Chlorophyll'], experimental=experimental_caro,
                                filename=f"{RESULTS_PATH}/pigments/chlorophyll_{condition}.png",
                                y_label="Macromolecule (g/gDW)", experimental_label=molecules)

    if 'Caro_concentration' in matrix.matrix[condition].columns:
        if condition.startswith("fachet") or condition.startswith("Xi") or condition.startswith("Yimei"):
            molecules = ['Caro_concentration']
            experimental_caro = [[matrix.matrix[condition][molecule].dropna().index.astype(float).tolist(),
                                  matrix.matrix[condition][molecule].dropna().tolist()] for molecule in molecules]
            plot_concentrations(concentrations, y=['Carotene_concentration'], experimental=experimental_caro,
                                filename=f"{RESULTS_PATH}/pigments/carotene_conc_{condition}.png",
                                y_label="Macromolecule (g/L)", experimental_label=molecules)

    if 'Chlorophyll_concentration' in matrix.matrix[condition].columns:
        if condition.startswith("fachet") or condition.startswith("Xi") or condition.startswith("Yimei"):
            molecules = ['Chlorophyll_concentration']
            experimental_caro = [[matrix.matrix[condition][molecule].dropna().index.astype(float).tolist(),
                                  matrix.matrix[condition][molecule].dropna().tolist()] for molecule in molecules]
            plot_concentrations(concentrations, y=['Chlorophyll_concentration'], experimental=experimental_caro,
                                filename=f"{RESULTS_PATH}/pigments/chlorophyll_conc_{condition}.png",
                                y_label="Macromolecule (g/L)", experimental_label=molecules)

    if 'NO3' in matrix.matrix[condition].columns:
        molecules = ['NO3']
        experimental_no3 = [[matrix.matrix[condition][molecule].dropna().index.astype(float).tolist(),
                             matrix.matrix[condition][molecule].dropna().tolist()] for molecule in molecules]
        plot_concentrations(concentrations, y=['Nitrate'], experimental=experimental_no3,
                            filename=f"{RESULTS_PATH}/concentrations/nitrate_{condition}.png", y_label="Macromolecule (g/gDW)",
                            experimental_label=molecules)

    plot_concentrations(concentrations, y=["Biomass", "ActiveBiomass"], experimental=experimental,
                        filename=f"{RESULTS_PATH}/biomass_concentrations/biomass_concentrations_{condition}.png",
                        y_label="Biomass (g/L)")

    plot_concentrations(concentrations, y=["Phosphate"], secondary_axis=["Nitrate"],
                        filename=f"{RESULTS_PATH}/concentrations/external_concentrations_{condition}.png",
                        y_label="Phosphate (mmol/L)", secondary_y_label="Nitrate (mmol/L)")

    if ("Glycerol" in matrix.matrix[condition].columns and "Starch" in matrix.matrix[condition].columns and "TAG" in
            matrix.matrix[condition].columns):
        plot_concentrations(concentrations, y=["Glycerol", "Starch", "TAG"],
                            filename=f"{RESULTS_PATH}/quotas/intracellular_quotas_{condition}.png",
                            y_label="Quota (g/gDW)", secondary_axis=["Carotene", "Chlorophyll"],
                            secondary_y_label="Quota (g/gDW)")

    # molecules = ["Protein", "Lipid", "Carbohydrate"]
    # if all(molecule in matrix.matrix[condition].columns for molecule in molecules):
    #     experimental = [[matrix.matrix[condition][molecule].dropna().index.astype(float).tolist(),
    #                      matrix.matrix[condition][molecule].dropna().tolist()] for molecule in molecules]
    #     plot_concentrations(concentrations, y=molecules, experimental=experimental,
    #                         filename=f"{DATA_PATH}/dfba/macros/macromolecules_{condition}.png",
    #                         y_label="Macromolecule (g/gDW)", experimental_label=molecules)

    concentrations.to_csv(f"{RESULTS_PATH}/concentrations/concentrations_{condition}.csv", index=False)
    trajectories.to_csv(f"{RESULTS_PATH}/trajectories/trajectories_{condition}.csv", index=False)


@timeout(40)
def create_dfba_model(condition, parameters, create_plots=False):
    if 'Time (d)' not in matrix.matrix[condition].columns:
        matrix.matrix[condition]['Time (d)'] = matrix.matrix[condition].index
    dfba_model = DfbaModel(fba_model)
    dfba_model.solver_data.set_display("none")
    dfba_model.solver_data.set_algorithm("direct")
    # dfba_model.solver_data.set_rel_tolerance(1e-3)
    # dfba_model.solver_data.set_ode_method("BDF")

    kinetic_vars = get_kinetic_variables()
    parameters.update(kinetic_vars)
    dfba_model.add_kinetic_variables(list(kinetic_vars.values()))

    exchange_fluxes = {}
    for key, value in get_exchange_fluxes().items():
        exchange_fluxes[key] = ExchangeFlux(value)

    dfba_model.add_exchange_fluxes(list(exchange_fluxes.values()))
    parameters.update(exchange_fluxes)
    parameters['starch_production'] = parameters['v_S'] * 48660.195 / 1000
    parameters['chl_production'] = parameters['v_chla'] * 894.49 / 1000 + parameters['v_chlb'] * 908.49 / 1000
    parameters['caro_production'] = parameters['v_C'] * 536.87 / 1000
    parameters['lutein_production'] = parameters['v_lutein'] * 568.87 / 1000
    parameters['tag_production'] = parameters['v_tag'] * 904.78 / 1000
    parameters['total_growth_rate'] = (parameters['mu'] +
                                       parameters['caro_production'] + parameters['chl_production'] + parameters['starch_production'] + parameters['tag_production'] +
                                       parameters['lutein_production'])

    for key, value in get_dynamic_expressions(parameters).items():
        dfba_model.add_rhs_expression(key, value)

    """
    Experiment-dependent
    """
    parameters["Eo"] = matrix.conditions["Light (umol/m^2.s)"].loc[condition]
    parameters["nacl"] = matrix.conditions["Salinity g/L"].loc[condition]
    parameters["Lr"] = matrix.conditions["Lr"].loc[condition]
    parameters["aeration"] = matrix.conditions["Aeration rate"].loc[condition]

    light_sources = matrix.conditions["Light sources"].loc[condition].split(",")
    for light_source in light_sources:
        fba_model.reactions.get_by_id(light_source).bounds = (0, 10000)

    """
    General parameters
    """
    parameters["q"] = parameters["n_quota"] / parameters["wNmax"]
    parameters["n"] = 1 - (parameters["q"] / (parameters["q"] + parameters["K_nitrogen_quota"]))
    x_storage = parameters["carotene"] + parameters["starch"]    + parameters["tag"] # + parameters["glycerol"]
    cell_size_increase = 1 / (1 - x_storage)
    parameters["z"] = (cell_size_increase - 1) / (parameters["t_max"] - 1)
    parameters["nitrogen_mass_quota"] = parameters["n_quota"] * 14.01 / 1000
    parameters["phosphate_mass_quota"] = parameters["p_quota"] * 30.97 / 1000

    """
    Light
    """
    parameters["Ex"], parameters["Ex0"] = get_bounds("light", parameters)
    dfba_model.add_exchange_flux_lb("EX_C00205__dra", sp.Max(sp.N(parameters["Ex"]), 0), parameters["light"])  #
    dfba_model.add_exchange_flux_ub("EX_C00205__dra", sp.Max(sp.N(parameters["Ex"]), 0), parameters["light"])
    """
    NO3
    """
    dfba_model.add_exchange_flux_lb("EX_C00244__dra", get_bounds("nitrate", parameters), parameters["nitrate"])  # 4.07
    # dfba_model.add_exchange_flux_lb("EX_C00244__dra", 1000, parameters["nitrate"])
    #     nitrate_quota = sp.Max(0, 1 - (4.8697 * F / X) / n_quota)
    dfba_model.add_exchange_flux_lb("DM_C00244__cytop", get_bounds("internal_nitrate", parameters))

    """
    HPO4
    """
    dfba_model.add_exchange_flux_lb("EX_C00009__dra", get_bounds("phosphate", parameters), parameters["phosphorus"])
    # dfba_model.add_exchange_flux_lb("EX_C00009__dra", 1000, parameters["phosphorus"])
    #     polyP_quota = sp.Max(0, 1 - (0.295 * F / X) / p_quota)
    dfba_model.add_exchange_flux_lb("DM_C00404__vacu", get_bounds("polyphosphate", parameters))

    """
    Starch
    """
    # dfba_model.add_exchange_flux_lb("DM_C00369__chlo", get_bounds("starch_consumption", parameters))
    dfba_model.add_exchange_flux_ub("DM_C00369__chlo", get_bounds("starch_production", parameters), parameters["starch"])

    """
    Carotene
    """
    dfba_model.add_exchange_flux_ub("DM_C02094__chlo", get_bounds("carotene", parameters), parameters["carotene"])

    """
    Lutein
    """
    dfba_model.add_exchange_flux_ub("DM_C08601__chlo", get_bounds("lutein", parameters), parameters["lutein"])

    """
    Chlorophyll
    """
    sum_chl = get_bounds("chlorophyll", parameters)
    dfba_model.add_exchange_flux_lb("DM_C05306__chlo", Abs(Min(sum_chl, 0)) * 1.73 / 2.73, parameters["chlorophyll"])
    dfba_model.add_exchange_flux_lb("DM_C05307__chlo", Abs(Min(sum_chl, 0)) / 2.73, parameters["chlorophyll"])
    dfba_model.add_exchange_flux_ub("DM_C05306__chlo", Max(sum_chl, 0) * 1.73 / 2.73, parameters["chlorophyll"])
    dfba_model.add_exchange_flux_ub("DM_C05307__chlo", Max(sum_chl, 0) / 2.73, parameters["chlorophyll"])
    """
    Glycerol
    """
    # wgly_max = 0.17  # https://doi.org/10.1016/j.biortech.2008.02.042
    #     dfba_model.add_exchange_flux_ub("DM_C00116__cytop", get_bounds("glycerol", parameters), parameters["glycerol"])

    """
    TAG
    """
    dfba_model.add_exchange_flux_ub("DM_C00422__lip", get_bounds("tag", parameters), parameters["tag"])

    """
    CO2
    """
    dfba_model.add_exchange_flux_lb("EX_C00011__dra", get_bounds("co2", parameters))

    """
    Simulate
    """
    dfba_model.add_initial_conditions(get_initial_conditions(matrix, condition))
    max_time = max(matrix.matrix[condition]['Time (d)'].astype(float).tolist()) + 1
    time_step = 1 / 72
    concentrations, trajectories = dfba_model.simulate(0.0, max_time, time_step,
                                                       ["e_ActiveBiomass__cytop", 'EX_C00009__dra',
                                                        "DM_C02094__chlo", "EX_C00244__dra",
                                                        "EX_C00011__dra",
                                                        "DM_C00404__vacu",
                                                        "DM_C00244__cytop", "EX_C00205__dra", "DM_C08601__chlo",
                                                        "DM_C05306__chlo", "DM_C05307__chlo",
                                                        "DM_C00369__chlo", "DM_C00422__lip"
                                                        ])
    if concentrations.time.tolist()[-1] < max_time/2:
        time_step = 1 / 48
        concentrations, trajectories = dfba_model.simulate(0.0, max_time, time_step,
                                                           ["e_ActiveBiomass__cytop", 'EX_C00009__dra',
                                                            "DM_C02094__chlo", "EX_C00244__dra",
                                                            "EX_C00011__dra",
                                                            "DM_C00404__vacu",
                                                            "DM_C00244__cytop", "EX_C00205__dra", "DM_C08601__chlo",
                                                            "DM_C05306__chlo", "DM_C05307__chlo",
                                                            "DM_C00369__chlo", "DM_C00422__lip"
                                                            ])

    active_biomass_fraction = concentrations['ActiveBiomass'] / concentrations['Biomass']
    concentrations.loc[:, 'Protein'] = abs(fba_model.reactions.e_ActiveBiomass__cytop.metabolites[
                                               fba_model.metabolites.e_Protein__cytop]) * active_biomass_fraction
    carbs = abs(fba_model.reactions.e_ActiveBiomass__cytop.metabolites[
                    fba_model.metabolites.e_Carbohydrate__cytop]) * active_biomass_fraction
    concentrations.loc[:, 'Carbohydrate'] = carbs
    polar_lipids = abs(fba_model.reactions.e_ActiveBiomass__cytop.metabolites[
                           fba_model.metabolites.e_Lipid__cytop]) * active_biomass_fraction
    concentrations.loc[:, 'Lipid'] = polar_lipids

    indexes = matrix.matrix[condition].index.astype(float)
    indexes = [e for e in indexes]
    experimental = [(indexes, matrix.matrix[condition]["DW"].tolist())]

    concentrations.loc[:, "Chlorophyll_concentration"] = concentrations['Chlorophyll'] * concentrations['Biomass']
    concentrations.loc[:, "Carotene_concentration"] = concentrations['Carotene'] * concentrations['Biomass']
    concentrations.loc[:, 'Lutein_concentration'] = concentrations['Lutein'] * concentrations['Biomass']

    if create_plots:
        generate_all_plots(condition, concentrations, experimental, trajectories)

    return concentrations, trajectories


def get_closest(list_a, list_b, return_index=False):
    """
    Gets the closest values from list_b to list_a.
    Parameters
    ----------
    list_a
    list_b

    Returns
    -------

    """
    closest_values = []
    indexes = []
    for num_a in list_a:
        index = np.abs(np.array(list_b) - num_a).argmin()
        closest_values.append(list_b[index])
        indexes.append(index)
    if return_index:
        return indexes
    return closest_values


def fitness(conditions_names, parameters_names, parameters_under_optimization=None, initial_parameters=None):
    """
    Calculates the fitness of a set of parameters.
    Parameters
    ----------
    parameters_names
    conditions_names
    parameters_under_optimization
    initial_parameters

    Returns
    -------

    """
    if not parameters_under_optimization:
        parameters = {}
        for i, parameter_name in enumerate(parameters_names):
            parameters[parameter_name] = 2 ** initial_parameters[i]
    else:
        parameters = json.load(open(f"{DATA_PATH}/parameters/initial_parameters.json", "r"))
        for i, parameter_name in enumerate(parameters_under_optimization):
            parameters[parameter_name] = 2 ** initial_parameters[i]
    try:
        total_error = sum(Parallel(n_jobs=len(conditions_names), timeout=30, backend="multiprocessing")(
            delayed(evaluate_trial)(parameters, condition=condition) for condition in
            conditions_names)) / len(conditions_names) * 100  # total_error = sum(progress_imap(partial(evaluate_trial, parameters, True), conditions_names,  #  process_timeout = 30))  #
    except Exception as e:
        print(e)
        total_error = 1e3
    # print(f"Total error from set of parameters: {total_error}")
    with open(f"{RESULTS_PATH}/logs/temp_error.log", "w") as file:
        file.write(f"{total_error}\n")
    return round(total_error, 2)


def fitness_func(initial_parameters, conditions_names, parameters_names, parameters_under_optimization=None):
    """
    Calculates the fitness of a set of parameters.
    Parameters
    ----------
    parameters_names
    conditions_names
    parameters_under_optimization
    initial_parameters

    Returns
    -------

    """
    if not parameters_under_optimization:
        parameters = {}
        for i, parameter_name in enumerate(parameters_names):
            parameters[parameter_name] = 2 ** initial_parameters[i]
    else:
        parameters = json.load(open(f"{DATA_PATH}/parameters/initial_parameters.json", "r"))
        for i, parameter_name in enumerate(parameters_under_optimization):
            parameters[parameter_name] = 2 ** initial_parameters[i]
    try:
        total_error = Parallel(n_jobs=len(conditions_names), timeout=30, backend="multiprocessing")(
            delayed(evaluate_trial)(parameters, condition=condition) for condition in
            conditions_names) #/ len(conditions_names) * 100  # total_error = sum(progress_imap(partial(evaluate_trial, parameters, True), conditions_names,  #  process_timeout = 30))  #
        # total_error = Parallel(n_jobs=len(conditions_names), timeout=30, backend="multiprocessing")(
        #     delayed(evaluate_trial)(parameters, condition=condition) for condition in
        #     conditions_names)
        error_1 = min(total_error)
        error_2 = max(total_error)
        # total_error = error_1*1 + error_2*10
        # minimum = min(total_error)
        # maximum = max(total_error)
        # if minimum >= 0.15:
        #     return (maximum*2,)
        # return (maximum, )
    except Exception as e:
        print(e)
        total_error = 1e3
    # print(f"Total error from set of parameters: {total_error}")
    with open(f"{RESULTS_PATH}/logs/temp_error.log", "w") as file:
        file.write(f"{total_error}\n")
    return (round(total_error, 3),)

def fitness_func_mo(initial_parameters, conditions_names, parameters_names, parameters_under_optimization=None):
    """
    Calculates the fitness of a set of parameters.
    Parameters
    ----------
    parameters_names
    conditions_names
    parameters_under_optimization
    initial_parameters

    Returns
    -------

    """
    if not parameters_under_optimization:
        parameters = {}
        for i, parameter_name in enumerate(parameters_names):
            try:
                parameters[parameter_name] = 2 ** initial_parameters[i]
            except OverflowError:
                return [(condition, 1e3) for condition in conditions_names]
    else:
        parameters = json.load(open(f"{DATA_PATH}/parameters/initial_parameters.json", "r"))
        for i, parameter_name in enumerate(parameters_under_optimization):
            try:
                parameters[parameter_name] = 2 ** initial_parameters[i]
            except OverflowError:
                print(f"OverflowError for parameter: {parameter_name}")
                parameters[parameter_name] = float('inf')
    try:
        total_error = Parallel(n_jobs=len(conditions_names), timeout=30, backend="multiprocessing")(
            delayed(evaluate_trial_mo)(parameters, condition=condition) for condition in
            conditions_names)  # total_error = sum(progress_imap(partial(evaluate_trial, parameters, True), conditions_names,  #  process_timeout = 30))  #
    except Exception as e:
        print("Error in fitness_func_mo:")
        print(e)
        total_error = [(condition, 1e3) for condition in conditions_names]
    # total_error is a list of tuples with the condition and the error. sort the tuples by the condition name
    total_error = sorted(total_error, key=lambda x: x[0])
    # print(f"Total error from set of parameters: {total_error}")
    with open(f"{RESULTS_PATH}/logs/temp_error_mo.log", "w") as file:
        file.write(f"{total_error}\n")
    sorted_errors = [e[1] for e in total_error]
    return tuple([round(e, 3) for e in sorted_errors])


def evaluate_trial(parameters, create_plots=False, condition=None, targets=None):
    """
    Evaluates a trial.
    Parameters
    ----------
    parameters (dict): Dictionary with the parameters to use in the model.
    create_plots (bool): Whether to create plots or not.
    condition (str): Condition to create the model for.

    Returns (float): The total error of the trial.
    -------

    """
    # print(f"Trial: {condition}\n")
    total_error = 0
    total_number_of_points = 1
    mat = matrix.matrix[condition]
    mat['Time (d)'] = [round(e, 2) for e in mat.index.astype(float)]
    try:
        concentrations, trajectories = create_dfba_model(condition, parameters, create_plots)
        # to_fit = {"Biomass": "DW", "Carotene": "Caro", "Chlorophyll": "Chl", "Starch": "Starch", "Nitrate": "NO3", 'Protein': 'Protein', 'Carbohydrate': 'Carbohydrate', 'Lipid': 'Lipid',
        #           "Chlorophyll_concentration": "Chlorophyll_concentration", "Carotene_concentration": "Caro_concentration"
        #           }  #
        to_fit = {
            "Biomass": "DW",
            # 'Lipid': 'Lipid', 'Protein': 'Protein', 'Carbohydrate': 'Carbohydrate',
            "Carotene": "Caro",
           # "Chlorophyll": "Chl",
            "Lutein": "Lutein",
            # "Chlorophyll_concentration": "Chlorophyll_concentration",
            # "Carotene_concentration": "Caro_concentration",
            # "Lutein_concentration": "Lutein_concentration"
        }
        to_fit = {key: value for key, value in to_fit.items() if key in targets}
        experimental_time = np.array(mat["Time (d)"])
        if concentrations.time.max() < experimental_time.max():
            return 1e3
        closest = get_closest(experimental_time, concentrations.time)
        at_time = concentrations.loc[concentrations.time.isin(closest)]
        at_time.reset_index(inplace=True, drop=True)
        mat.reset_index(inplace=True, drop=True)
        for simulation_name, experimental_name in to_fit.items():
            if experimental_name in mat.columns:
                experimental = mat[experimental_name]
                simulated = at_time[simulation_name]
                relative_error, number_of_points = get_rmse(experimental, simulated)
                total_error += relative_error
                total_number_of_points += number_of_points
                # print(f"Total error for {simulation_name}:\n{total_error}")
    except Exception as e:
        print(e)
        with open(f"{RESULTS_PATH}/logs/temp_error.log", "a") as file:
            file.write(f"{e}\n")
        total_error = 1e3
    return round(total_error, 3)

def evaluate_trial_mo(parameters, create_plots=False, condition=None):
    """
    Evaluates a trial.
    Parameters
    ----------
    parameters (dict): Dictionary with the parameters to use in the model.
    create_plots (bool): Whether to create plots or not.
    condition (str): Condition to create the model for.

    Returns (float): The total error of the trial.
    -------

    """
    # print(f"Trial: {condition}\n")
    total_error = 0
    total_number_of_points = 1
    mat = matrix.matrix[condition]
    mat['Time (d)'] = [round(e, 2) for e in mat.index.astype(float)]
    try:
        concentrations, trajectories = create_dfba_model(condition, parameters, create_plots)
        # to_fit = {"Biomass": "DW", "Carotene": "Caro", "Chlorophyll": "Chl", "Starch": "Starch", "Nitrate": "NO3", 'Protein': 'Protein', 'Carbohydrate': 'Carbohydrate', 'Lipid': 'Lipid',
        #           "Chlorophyll_concentration": "Chlorophyll_concentration", "Carotene_concentration": "Caro_concentration"
        #           }  #
        to_fit = {
            # "Biomass": "DW",
            # 'Lipid': 'Lipid', 'Protein': 'Protein', 'Carbohydrate': 'Carbohydrate',
            # "Carotene": "Caro",
           # "Chlorophyll": "Chl",
            "Lutein": "Lutein",
            # "Chlorophyll_concentration": "Chlorophyll_concentration",
            # "Carotene_concentration": "Caro_concentration",
            # "Lutein_concentration": "Lutein_concentration"
        }
        experimental_time = np.array(mat["Time (d)"])
        if concentrations.time.max() < experimental_time.max():
            return condition, 1e3
        closest = get_closest(experimental_time, concentrations.time)
        at_time = concentrations.loc[concentrations.time.isin(closest)]
        at_time.reset_index(inplace=True, drop=True)
        mat.reset_index(inplace=True, drop=True)
        for simulation_name, experimental_name in to_fit.items():
            if experimental_name in mat.columns:
                experimental = mat[experimental_name]
                simulated = at_time[simulation_name]
                relative_error, number_of_points = get_rmse(experimental, simulated)
                total_error += relative_error
                total_number_of_points += number_of_points
                # print(f"Total error for {simulation_name}:\n{total_error}")
    except Exception as e:
        print(e)
        with open(f"{RESULTS_PATH}/logs/temp_error.log", "a") as file:
            file.write(f"{e}\n")
        total_error = 1e3
    return condition, round(total_error, 3)


def get_relative_error(experimental, simulated):
    """
    Calculates the relative error between two vectors.
    Parameters
    ----------
    experimental
    simulated

    Returns
    -------

    """
    relative_error = 0
    number_of_points = len(experimental)
    if simulated.shape[0] < experimental.shape[0]:
        return 1000, number_of_points
    try:
        experimental.dropna(inplace=True)
        intersection = [e for e in simulated.index if e in experimental.index]
        simulated = simulated.loc[intersection]
        experimental = experimental.loc[intersection]
        abs_error = np.abs(experimental - simulated)
        number_of_points = len(experimental)
        with np.errstate(divide='ignore', invalid='ignore'):
            relative_error = round(sum(np.where(experimental == 0, np.inf, abs_error / np.abs(experimental))), 4)
    #         print(f"Relative error:\n{relative_error}")
    except Exception as e:
        print(e)
        print(f"Experimental values:\n{experimental}")
        print(f"Simulated values:\n{simulated}")
    if not isinstance(relative_error, numbers.Number):
        print(f"Relative error:\n{relative_error}")
        print(f"Experimental values:\n{experimental}")
        print(f"Simulated values:\n{simulated}")
    if np.isnan(relative_error):
        print("NaN found!!")
        print(f"Experimental values:\n{experimental}")
        print(f"Simulated values:\n{simulated}")
        relative_error = 100
    return relative_error, number_of_points


import numpy as np
import numbers


def get_rmse(experimental, simulated):
    """
    Calculates the RMSE between two vectors.

    Parameters
    ----------
    experimental: pandas Series
    simulated: pandas Series

    Returns
    -------
    rmse: float
        The root mean squared error between experimental and simulated data.
    number_of_points: int
        The number of points used in the calculation.
    """
    rmse = 0
    number_of_points = len(experimental)
    if simulated.shape[0] < experimental.shape[0]:
        return 1000, number_of_points

    try:
        # Remove NaN values from experimental data
        experimental.dropna(inplace=True)

        # Intersect indices to ensure both datasets match
        intersection = [e for e in simulated.index if e in experimental.index]
        simulated = simulated.loc[intersection]
        experimental = experimental.loc[intersection]

        # Calculate the squared error
        squared_error = np.square((experimental - simulated) / experimental)

        # Calculate RMSE
        rmse = np.sqrt(squared_error.mean())

        # Update number of points
        number_of_points = len(experimental)

    except Exception as e:
        print(e)
        print(f"Experimental values:\n{experimental}")
        print(f"Simulated values:\n{simulated}")

    # Handle any potential errors or non-number results
    if not isinstance(rmse, numbers.Number):
        print(f"RMSE:\n{rmse}")
        print(f"Experimental values:\n{experimental}")
        print(f"Simulated values:\n{simulated}")

    if np.isnan(rmse):
        print("NaN found!!")
        print(f"Experimental values:\n{experimental}")
        print(f"Simulated values:\n{simulated}")
        rmse = 100

    return rmse, number_of_points


def callback_f(pbar, x):
    """
    Callback function for the optimization.
    Parameters
    ----------
    pbar
    x

    Returns
    -------

    """
    with open(f"{RESULTS_PATH}/logs/temp_error.log") as file:
        error = float(file.read())
    pbar.set_description(f"Current objective: {round(error, 3)}")
    pbar.update(1)

def decode_solution(population):
    result = []
    for i, individual in enumerate(population):
        try:
            result.append([2 ** round(e, 5) if round(e, 5) != 0 else e for e in individual])
        except OverflowError:
            result.append([float('inf') if round(e, 5) != 0 else e for e in individual])
    return result

def parameter_optimization_ea(custom_parameters: list = None):
    """
    Runs the parameter optimization.
    Returns
    -------

    """

    def check_bounds(individual, bounds):
        for i in range(len(individual)):
            if individual[i] < bounds[i][0]:
                individual[i] = bounds[i][0]
            elif individual[i] > bounds[i][1]:
                individual[i] = bounds[i][1]
        return individual
    from deap import base, creator, tools

    initial_parameters = OrderedDict(json.load(open(f"{DATA_PATH}/parameters/initial_parameters.json", "r")))
    old_parameters = copy.deepcopy(initial_parameters)
    bounds_ordered_dict = OrderedDict(json.load(open(f"{DATA_PATH}/parameters/parameters_bounds.json", "r")))
    bounds_ordered_dict = OrderedDict({key: bounds_ordered_dict[key] for key in initial_parameters.keys()})
    conditions_names = set(matrix.matrix.keys()) - {"Resume"}

    conditions_names = tuple(e for e in conditions_names if e.startswith("OC") or e.startswith("SC")) #
    # to_ignore = ["fachet", "Xi", "Yimei", "22"]
    # conditions_names = list(set(sorted([e for e in conditions_names if not any(e.startswith(ignore) for ignore in to_ignore)]) + ["OC", "SC"]))

    # validation = select_random_conditions(list(conditions_names), 5, [])

    initial_error = sum(
        progress_imap(partial(evaluate_trial, initial_parameters, True), conditions_names, n_cpu=len(conditions_names)))
    print(f"Initial error: {initial_error}")
    # with open(f"{RESULTS_PATH}/validation.txt", 'w') as f:
    #     f.write("Validation conditions:\n")
    #     for e in validation:
    #         f.write(f"{e}\n")
    #     f.write(f"Initial error was: {initial_error}")
    shutil.make_archive(f'{RESULTS_PATH}', 'zip', f'{RESULTS_PATH}')

    if custom_parameters:
        initial_parameters = {key: initial_parameters[key] for key in custom_parameters}
        bounds_ordered_dict = OrderedDict({key: bounds_ordered_dict[key] for key in custom_parameters})

    def random_float_within_bounds(bounds):
        return [np.random.uniform(low, high) for low, high in bounds]

    initial_parameters_log = [math.log2(e) for e in initial_parameters.values()]
    bounds_log = [(math.log2(e[0] + 1e-10), math.log2(e[1] + 1e-10)) for e in bounds_ordered_dict.values()]

    def create_individual():
        return creator.Individual(list(initial_parameters_log))

    # DEAP requires a special Fitness class and an Individual class
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # Create a toolbox for our evolutionary algorithm
    toolbox = base.Toolbox()
    import multiprocessing

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    toolbox.register("attr_float", random_float_within_bounds, list(bounds_log))
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register the fitness function
    toolbox.register("evaluate", fitness_func, conditions_names=conditions_names, parameters_names=list(initial_parameters.keys()), parameters_under_optimization=custom_parameters)

    # Register the crossover operator (here using blend crossover)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)

    # Register the mutation operator (here using Gaussian mutation)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)

    # Register the selection operator (here using tournament selection)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Genetic Algorithm parameters
    population_size = 150
    crossover_probability = 0.8
    mutation_probability = 0.3
    generations = 50

    # Create the population
    population = toolbox.population(n=population_size)
    for i in range(5):  # seed 5 individuals with initial guess
        population[i] = create_individual()
    best_fitness = 1e3

    # Run the optimization with tqdm progress bar
    with tqdm(total=generations, desc=f"Running optimization") as pbar:
        for gen in range(generations):
            # Select the next generation individuals
            offspring = toolbox.select(population, len(population))
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.random() < crossover_probability:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if np.random.random() < mutation_probability:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # for ind in offspring:
            #     check_bounds(ind, bounds_log)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Replace the current population with the offspring
            population[:] = offspring

            if best_fitness > min([ind.fitness.values[0] for ind in population]):
                best_fitness = min([ind.fitness.values[0] for ind in population])
                pbar.set_description(f"Current objective: {round(best_fitness, 3)}")
            # Update the progress bar
            pbar.update(1)

    # Extracting the best individual
    best_ind = tools.selBest(population, 1)[0]
    optimal_fitness = best_ind.fitness.values[0]
    print(f"Best individual is: {best_ind}, with fitness: {optimal_fitness}")
    pool.close()
    optimal_params = [2 ** e for e in best_ind]
    # optimal_params = [e for e in best_ind]
    for index, param in enumerate(optimal_params):
        if round(param, 5) != 0:
            optimal_params[index] = round(param, 5)

    optimal_parameters = {list(initial_parameters.keys())[index]: param for index, param in enumerate(optimal_params)}
    # Print optimized parameters and fitness value
    print("Optimized Parameters:\n")

    for key, value in old_parameters.items():
        if key not in optimal_parameters:
            optimal_parameters[key] = value

    with open(f"{RESULTS_PATH}/parameters/optimized_parameters.json", "w") as f:
        json.dump(optimal_parameters, f, indent=4)

    with open(f"{RESULTS_PATH}/parameters/optimized_parameters.txt", "w") as f:
        for index, param in enumerate(optimal_params):
            f.write(f"{list(initial_parameters.keys())[index]}\t{param}\n")
            print(f"{list(initial_parameters.keys())[index]}:\t{param}\t{list(initial_parameters.values())[index]}\n")
        f.write(f"Error after optimization:\t{str(optimal_fitness)}")
    print("Optimized Fitness Value: ", optimal_fitness)
    final_error = sum(
        progress_imap(partial(evaluate_trial, optimal_parameters, True), conditions_names, n_cpu=len(conditions_names))) / len(conditions_names) * 100
    print(f"Final error was: {final_error}")

    # with tqdm(total=len(validation), desc="Running validation") as pbar:
    #     validation_error = sum(Parallel(n_jobs=30)(
    #         delayed(evaluate_trial)(optimal_parameters, create_plots=True, condition=condition) for condition in
    #         validation))
    # pbar.set_description(f"Validation error: {validation_error}")
    # # validation_error = sum(progress_imap(partial(evaluate_trial, fba_model, matrix, optimal_parameters, True), conditions_names, n_cpu=len(conditions_names)))
    # print(f"Validation error was: {validation_error}")
    # with open(f"{RESULTS_PATH}/validation.txt", 'a') as f:
    #     f.write(f"\nValidation error was: {validation_error}")

    # run_all_parallel(optimal_parameters)
    # add non-cuistom parameters

    run_condition("OC", optimal_parameters)
    run_condition("SC", optimal_parameters)


def parameter_optimization_ea_mo(custom_parameters: list = None):
    """
    Runs the multi-objective parameter optimization using DEAP.
    """
    from deap import base, creator, tools
    import numpy as np
    import math
    from collections import OrderedDict
    from tqdm import tqdm
    import copy
    import json
    from multiprocessing import Pool
    seed = 42
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Load parameters
    initial_parameters = OrderedDict(json.load(open(f"{DATA_PATH}/parameters/initial_parameters.json", "r")))
    old_parameters = copy.deepcopy(initial_parameters)
    bounds_ordered_dict = OrderedDict(json.load(open(f"{DATA_PATH}/parameters/parameters_bounds.json", "r")))
    bounds_ordered_dict = OrderedDict({key: bounds_ordered_dict[key] for key in initial_parameters.keys()})
    conditions_names = set(matrix.matrix.keys()) - {"Resume"}
    # validation = select_random_conditions(list(conditions_names), 5, ['fachet_HLND', "Yimei_HL"])
    conditions_names = tuple(e for e in conditions_names if e.startswith("OC") or  e.startswith("SC")) #

    if custom_parameters:
        initial_parameters = {key: initial_parameters[key] for key in custom_parameters}
        bounds_ordered_dict = OrderedDict({key: bounds_ordered_dict[key] for key in custom_parameters})

    # Log normalization of initial parameters
    initial_parameters_log = [math.log2(e) for e in initial_parameters.values()]
    bounds_log = [(math.log2(e[0] + 1e-10), math.log2(e[1] + 1e-10)) for e in bounds_ordered_dict.values()]

    def create_individual():
        return creator.Individual(list(initial_parameters_log))

    # Step 1: Create a multi-objective fitness class with weights
    # Here we are assuming two objectives, one to minimize and one to maximize.
    # Adjust weights as needed, where each weight corresponds to an objective:
    # (-1.0, -1.0) for both minimization or (1.0, -1.0) for maximizing the first, minimizing the second.

    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -10.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    # Step 2: Create the toolbox for our evolutionary algorithm
    toolbox = base.Toolbox()
    pool = Pool()
    toolbox.register("map", pool.map)

    best_individuals = {}

    # Random initialization within log-bounds
    def random_float_within_bounds(bounds):
        return [np.random.uniform(low, high) for low, high in bounds]

    toolbox.register("attr_float", random_float_within_bounds, list(bounds_log))
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Step 3: Modify the fitness function to return multiple objectives
    # Assuming fitness_func can return a tuple (obj1, obj2)
    # where obj1 is to be minimized and obj2 is to be maximized (or both minimized)

    toolbox.register("evaluate", fitness_func_mo, conditions_names=conditions_names, parameters_names=list(initial_parameters.keys()), parameters_under_optimization=custom_parameters)

    # Step 4: Crossover, mutation, and selection operators
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)

    # Multi-objective selection operator using NSGA-II
    toolbox.register("select", tools.selNSGA2)

    # Step 5: Genetic Algorithm parameters
    population_size = 250
    crossover_probability = 0.8
    mutation_probability = 0.3
    generations = 50

    # Create the population
    population = toolbox.population(n=population_size)
    for i in range(5):  # Seed 5 individuals with initial guess
        population[i] = create_individual()
    best_fitness = [float('inf'), float('inf')]
    best_fitness_counter = 0
    open(f"{RESULTS_PATH}/logs/best_parameters.json", "w").close()
    open(f"{RESULTS_PATH}/logs/best_parameters.json", "w").close()
    # Step 6: Run the multi-objective optimization with progress bar
    with tqdm(total=generations, desc="Running optimization") as pbar:
        for gen in range(generations):
            offspring = toolbox.select(population, len(population))  # NSGA-II selects individuals
            offspring = list(map(toolbox.clone, offspring))

            # Apply crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.random() < crossover_probability:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if np.random.random() < mutation_probability:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            for ind in offspring:
                fit = ind.fitness.values
                if fit[1] < best_fitness[1]:
                    best_fitness[0] = fit[0]  # Update first objective
                    best_fitness[1] = fit[1]  # Update second objective
                    with open(f"{RESULTS_PATH}/logs/best_parameters.json", "a") as file:
                        res = decode_solution([ind])
                        best_fitness_counter += 1
                        key = str(best_fitness_counter) + "_" + str(fit[0]) + "_" + str(fit[1])
                        new_dict = {key: res}
                        json.dump(new_dict, file, indent=4)
                    best_individuals[key] = res


            # Replace the current population with the offspring
            population[:] = offspring

            # Update the progress bar with the best fitness
            pbar.set_description(f"Current best: Obj1: {round(best_fitness[0], 3)}, Obj2: {round(best_fitness[1], 3)}")            # Update progress bar
            pbar.update(1)

    with open(f"{RESULTS_PATH}/logs/best_individuals.json", "a") as file:
        json.dump(best_individuals, file, indent=4)

    # remove population with infinities
    population = [ind for ind in population if not any(e == float('inf') for e in ind.fitness.values)]

    # pareto_fronts = tools.sortNondominated(population, len(population), first_front_only=True)

    # Return the Pareto front for analysis
    pool.close()

    with open(f"{RESULTS_PATH}/parameters/pareto_front.json", "w") as f:
        for pareto_front in population:
            pareto_front_decoded = decode_solution([pareto_front])
            pareto_front_decoded = [ind for ind in pareto_front_decoded if not any(e == float('inf') for e in ind)]
            pareto_results = OrderedDict()
            for i, individual in enumerate(pareto_front_decoded):
                pareto_results[i] = (fitness_func_mo(pareto_front, conditions_names, list(initial_parameters.keys()), custom_parameters),
                    OrderedDict({key: individual[index] for index, key in enumerate(initial_parameters.keys())}))
                json.dump(pareto_results, f, indent=4)

    return pareto_results

def get_ae(df, col1, col2):
    """Calculate APE between two columns in a pandas DataFrame."""
    return round(np.abs(df[col1] - df[col2]) / df[col1], 3)

def get_rmse_v2(df, col1, col2):
    """Calculate RMSE between two columns in a pandas DataFrame."""
    return round(np.sqrt(((df[col1] - df[col2]) ** 2).mean()), 3)

def get_r2(df, col1, col2):
    """Calculate R^2 between two columns in a pandas DataFrame."""
    return round(r2_score(df[col1], df[col2]), 3)


def generate_trials_plots():
    """
    Generates plots for all trials
    Returns
    -------

    """
    (data_carotene, data_carotene_conc, data_chl, data_protein, data_lipid, data_carbohydrate, data_lutein,
     data_lipid_conc, data_protein_conc, data_carbohydrate_conc, data_biomass) = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    (experimental_data_carotene, experimental_data_carotene_concentration, experimental_data_chl,
     experimental_data_biomass, experimental_data_protein, experimental_data_lipid, experimental_data_carbohydrate, experimental_data_lutein, experimental_data_protein_concentration, experimental_data_carbohydrate_concentration,
     experimental_data_lipid_concentration, sd_carotene, sd_lutein, sd_chl, sd_carotene_concentration, sd_lutein_concentration, sd_chl_concentration,
     experimental_data_lutein_concentration, data_lutein_conc, data_chl_conc, experimental_data_chl_concentration) = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    for condition in matrix.conditions.index:
        if 'Caro' in matrix.matrix[condition].columns and not condition.startswith(
                "fachet") and not condition.startswith("Xi") and not condition.startswith("Yimei") and not {condition}.issubset({"OC", "SC", "OC_Sousa2024", "SC_Sousa2024"}):
            temp = pd.read_csv(f"{RESULTS_PATH}/concentrations/concentrations_{condition}.csv")
            final_time_index = get_closest(matrix.matrix[condition].index.astype(float).tolist(), temp.time.tolist(), return_index=True)[-1]
            dry_weight = matrix.matrix[condition]['DW'].dropna().tolist()[-1]
            data_biomass[condition] = temp.iloc[final_time_index]['Biomass']
            data_carotene[condition] = temp.iloc[final_time_index]['Carotene'] *1000
            data_carotene_conc[condition] = temp.iloc[final_time_index]['Carotene'] * temp.iloc[final_time_index]['Biomass'] *1000
            data_chl_conc[condition] = temp.iloc[final_time_index]['Chlorophyll'] * temp.iloc[final_time_index]['Biomass']
            data_lutein_conc[condition] = temp.iloc[final_time_index]['Lutein'] * temp.iloc[final_time_index]['Biomass'] *1000
            data_lutein[condition] = temp.iloc[final_time_index]['Lutein'] *1000
            data_chl[condition] = temp.iloc[final_time_index]['Chlorophyll']
            data_protein[condition] = temp.iloc[final_time_index]['Protein']
            data_protein_conc[condition] = temp.iloc[final_time_index]['Protein'] * temp.iloc[-1]['Biomass']
            data_lipid[condition] = temp.iloc[final_time_index]['Lipid']
            data_lipid_conc[condition] = temp.iloc[final_time_index]['Lipid'] * temp.iloc[final_time_index]['Biomass']
            data_carbohydrate[condition] = temp.iloc[final_time_index]['Carbohydrate']
            data_carbohydrate_conc[condition] = temp.iloc[final_time_index]['Carbohydrate'] * temp.iloc[final_time_index]['Biomass']
            experimental_data_carotene[condition] = matrix.matrix[condition]['Caro'].dropna().tolist()[-1] *1000
            experimental_data_lutein[condition] = matrix.matrix[condition]['Lutein'].dropna().tolist()[-1] *1000
            experimental_data_carotene_concentration[condition] = matrix.matrix[condition]['Caro'].dropna().tolist()[-1] * dry_weight *1000
            # experimental_data_chl[condition] = matrix.matrix[condition]['Chl'].dropna().tolist()[-1]
            # experimental_data_chl_concentration[condition] = matrix.matrix[condition]['Chl'].dropna().tolist()[-1] * dry_weight
            experimental_data_lutein_concentration[condition] = matrix.matrix[condition]['Lutein'].dropna().tolist()[-1] * dry_weight *1000
            sd_carotene_concentration[condition] = matrix.matrix[condition]['Caro_c_sd'].dropna().tolist()[-1] *1000
            sd_lutein_concentration[condition] = matrix.matrix[condition]['Lutein_c_sd'].dropna().tolist()[-1] *1000
            # sd_chl_concentration[condition] = matrix.matrix[condition]['Chl_c_sd'].dropna().tolist()[-1]
            sd_carotene[condition] = matrix.matrix[condition]['Caro_sd'].dropna().tolist()[-1] *1000
            sd_lutein[condition] = matrix.matrix[condition]['Lutein_sd'].dropna().tolist()[-1] *1000
            # sd_chl[condition] = matrix.matrix[condition]['Chl_sd'].dropna().tolist()[-1]

            data_protein[condition] = temp.iloc[-1]['Protein']
            data_lipid[condition] = temp.iloc[-1]['Lipid']
            data_lipid_conc[condition] = temp.iloc[-1]['Lipid'] * temp.iloc[-1]['Biomass']
            data_carbohydrate[condition] = temp.iloc[-1]['Carbohydrate']

            experimental_data_biomass[condition] = matrix.matrix[condition]['DW'].dropna().tolist()[-1]

            molecules = ["Protein", "Lipid", "Carbohydrate"]
            if all(molecule in matrix.matrix[condition].columns for molecule in molecules):
                experimental_data_protein[condition] = matrix.matrix[condition]['Protein'].dropna().tolist()[-1]
                experimental_data_protein_concentration[condition] = matrix.matrix[condition]['Protein'].dropna().tolist()[-1] * matrix.matrix[condition]['DW'].dropna().tolist()[-1]
                experimental_data_lipid[condition] = matrix.matrix[condition]['Lipid'].dropna().tolist()[-1]
                experimental_data_lipid_concentration[condition] = matrix.matrix[condition]['Lipid'].dropna().tolist()[-1] * matrix.matrix[condition]['DW'].dropna().tolist()[-1]
                experimental_data_carbohydrate[condition] = matrix.matrix[condition]['Carbohydrate'].dropna().tolist()[-1]
                experimental_data_carbohydrate_concentration[condition] = matrix.matrix[condition]['Carbohydrate'].dropna().tolist()[-1] * matrix.matrix[condition]['DW'].dropna().tolist()[-1]

    generate_plot_for_data(f"{RESULTS_PATH}/pigments/carotene_in_house.pdf", experimental_data_carotene, data_carotene, sd_carotene_concentration, r"$\beta$-Carotene (mg/gDW)")
    generate_plot_for_data(f"{RESULTS_PATH}/pigments/carotene_concentration_in_house.pdf", experimental_data_carotene_concentration, data_carotene_conc, sd_carotene_concentration, r"$\beta$-Carotene ($mg \cdot L^{-1}$)")
    generate_plot_for_data(f"{RESULTS_PATH}/pigments/lutein_in_house.pdf", experimental_data_lutein, data_lutein, sd_lutein, r"Lutein (mg/gDW)")
    generate_plot_for_data(f"{RESULTS_PATH}/pigments/lutein_concentration_in_house.pdf", experimental_data_lutein_concentration, data_lutein_conc, sd_lutein_concentration, r"Lutein ($mg \cdot L^{-1}$)")
    # generate_plot_for_data(f"{DATA_PATH}/dfba/pigments/chl_in_house.png", experimental_data_chl, data_chl, sd_chl, r"Chlorophyll (g/gDW)")
    # generate_plot_for_data(f"{DATA_PATH}/dfba/pigments/chl_concentration_in_house.png", experimental_data_chl_concentration, data_chl_conc, sd_chl_concentration, r"Chlorophyll (g/L)")

    generate_plot_for_data(f"{RESULTS_PATH}/macros/protein_in_house.png", experimental_data_protein, data_protein, {}, r"Protein (g/gDW)")
    generate_plot_for_data(f"{RESULTS_PATH}/macros/protein_concentration_in_house.png", experimental_data_protein_concentration, data_protein_conc, {}, r"Protein (g/L)")
    generate_plot_for_data(f"{RESULTS_PATH}/macros/lipid_in_house.png", experimental_data_lipid, data_lipid, {}, r"Lipid (g/gDW)")
    generate_plot_for_data(f"{RESULTS_PATH}/macros/lipid_concentration_in_house.png", experimental_data_lipid_concentration, data_lipid_conc, {}, r"Lipid (g/L)")
    generate_plot_for_data(f"{RESULTS_PATH}/macros/carbohydrate_in_house.png", experimental_data_carbohydrate, data_carbohydrate, {}, r"Carbohydrate (g/gDW)")
    generate_plot_for_data(f"{RESULTS_PATH}/macros/carbohydrate_concentration_in_house.png", experimental_data_carbohydrate_concentration, data_carbohydrate_conc, {}, r"Carbohydrate (g/L)")

    # merge expreimental data and simulated data of carotene and lutein and biomass into a single dataframe, where each row is a condition

    df = pd.DataFrame()
    df.index = matrix.conditions.index
    df['N mM'] = df.index.map({condition: matrix.matrix["Resume"].loc[condition]['[N] mmol'] for condition in matrix.conditions.index})
    df['P mM'] = df.index.map({condition: matrix.matrix["Resume"].loc[condition]['[P] mmol'] for condition in matrix.conditions.index})
    # df['Salinity'] = df.index.map({condition: matrix.matrix["Resume"].loc[condition]['Salinity g/L'] for condition in matrix.conditions.index})
    df['Aeration rate'] = df.index.map({condition: matrix.matrix["Resume"].loc[condition]['Aeration rate'] for condition in matrix.conditions.index})
    # df['Biomass'] =  df.index.map(experimental_data_biomass)
    # df['Biomass_simulated'] = df.index.map(data_biomass)
    # df['Biomass diff'] = get_ae(df, 'Biomass', 'Biomass_simulated')
    # df['Carotene'] = pd.Series(experimental_data_carotene)*1000
    # df['Carotene_simulated'] = pd.Series(data_carotene)*1000
    # df['Carotene_diff'] = get_ae(df, 'Carotene', 'Carotene_simulated')
    df['Carotene_concentration'] = pd.Series(experimental_data_carotene_concentration)
    df['Carotene_concentration_simulated'] = pd.Series(data_carotene_conc)
    df['Carotene_concentration_diff'] = get_ae(df, 'Carotene_concentration', 'Carotene_concentration_simulated')
    # df['Lutein'] = pd.Series(experimental_data_lutein)*1000
    # df['Lutein_simulated'] = pd.Series(data_lutein)*1000
    # df['Lutein_diff'] = get_ae(df, 'Lutein', 'Lutein_simulated')
    df['Lutein_concentration'] = pd.Series(experimental_data_lutein_concentration)
    df['Lutein_concentration_simulated'] = pd.Series(data_lutein_conc)
    df['Lutein_concentration_diff'] = get_ae(df, 'Lutein_concentration', 'Lutein_concentration_simulated')
    df.dropna(inplace=True)
    df.columns = [col.replace("_", " ") for col in df.columns]
    df.to_csv(f"{RESULTS_PATH}/pigments/pigments_matrix.tsv", sep="\t")
    df['N mM'] = df['N mM'].map(lambda x: f"{x:.1f}" if '.' in str(x) else f"{x:.0f}")  # Keeps 3 decimals max
    df['P mM'] = df['P mM'].map(lambda x: f"{x:.2f}" if '.' in str(x) else f"{x:.0f}")
    # df['Salinity'] = df['Salinity'].map(lambda x: f"{x:.0f}" if '.' in str(x) else f"{x:.0f}")
    df['Aeration rate'] = df['Aeration rate'].map(lambda x: f"{x:.0f}" if '.' in str(x) else f"{x:.0f}")
    # df['Biomass'] = df['Biomass'].map(lambda x: f"{x:.3f}" if '.' in str(x) else f"{x:.0f}")
    # df['Biomass simulated'] = df['Biomass simulated'].map(lambda x: f"{x:.3f}" if '.' in str(x) else f"{x:.0f}")
    # df['Biomass diff'] = df['Biomass diff'].map(lambda x: f"{x:.3f}" if '.' in str(x) else f"{x:.0f}")
    # df['Carotene'] = df['Carotene'].map(lambda x: f"{x:.3f}" if '.' in str(x) else f"{x:.0f}")
    # df['Carotene simulated'] = df['Carotene simulated'].map(lambda x: f"{x:.3f}" if '.' in str(x) else f"{x:.0f}")
    # df['Carotene diff'] = df['Carotene diff'].map(lambda x: f"{x:.3f}" if '.' in str(x) else f"{x:.0f}")
    df['Carotene concentration'] = df['Carotene concentration'].map(lambda x: f"{x:.3f}" if '.' in str(x) else f"{x:.0f}")
    df['Carotene concentration simulated'] = df['Carotene concentration simulated'].map(lambda x: f"{x:.3f}" if '.' in str(x) else f"{x:.0f}")
    df['Carotene concentration diff'] = df['Carotene concentration diff'].map(lambda x: f"{x:.3f}" if '.' in str(x) else f"{x:.0f}")
    # df['Lutein'] = df['Lutein'].map(lambda x: f"{x:.3f}" if '.' in str(x) else f"{x:.0f}")
    # df['Lutein simulated'] = df['Lutein simulated'].map(lambda x: f"{x:.3f}" if '.' in str(x) else f"{x:.0f}")
    # df['Lutein diff'] = df['Lutein diff'].map(lambda x: f"{x:.3f}" if '.' in str(x) else f"{x:.0f}")
    df['Lutein concentration'] = df['Lutein concentration'].map(lambda x: f"{x:.3f}" if '.' in str(x) else f"{x:.0f}")
    df['Lutein concentration simulated'] = df['Lutein concentration simulated'].map(lambda x: f"{x:.3f}" if '.' in str(x) else f"{x:.0f}")
    df['Lutein concentration diff'] = df['Lutein concentration diff'].map(lambda x: f"{x:.3f}" if '.' in str(x) else f"{x:.0f}")

    df.to_latex(f"{RESULTS_PATH}/pigments/pigments_matrix.tex", column_format="l" + "c" * len(df.columns), escape=False, longtable=True)

    # df['Carotene'] = df['Carotene'].astype(float)
    # df['Carotene simulated'] = df['Carotene simulated'].astype(float)
    df['Carotene concentration'] = df['Carotene concentration'].astype(float)
    df['Carotene concentration simulated'] = df['Carotene concentration simulated'].astype(float)
    # df['Lutein'] = df['Lutein'].astype(float)
    # df['Lutein simulated'] = df['Lutein simulated'].astype(float)
    df['Lutein concentration'] = df['Lutein concentration'].astype(float)
    df['Lutein concentration simulated'] = df['Lutein concentration simulated'].astype(float)
    # df['Biomass'] = df['Biomass'].astype(float)
    # df['Biomass simulated'] = df['Biomass simulated'].astype(float)

    # print(f"Carotene RMSE: {get_rmse_v2(df, 'Carotene', 'Carotene simulated')}")
    # print(f"Lutein RMSE: {get_rmse_v2(df, 'Lutein', 'Lutein simulated')}")
    print(f"Carotene concentration RMSE: {get_rmse_v2(df, 'Carotene concentration', 'Carotene concentration simulated')}")
    print(f"Lutein concentration RMSE: {get_rmse_v2(df, 'Lutein concentration', 'Lutein concentration simulated')}")
    # print(f"Biomass RMSE: {get_rmse_v2(df, 'Biomass', 'Biomass simulated')}")
    # print(f"Carotene R2: {get_r2(df, 'Carotene', 'Carotene simulated')}")
    # print(f"Lutein R2: {get_r2(df, 'Lutein', 'Lutein simulated')}")
    print(f"Carotene concentration R2: {get_r2(df, 'Carotene concentration', 'Carotene concentration simulated')}")
    print(f"Lutein concentration R2: {get_r2(df, 'Lutein concentration', 'Lutein concentration simulated')}")
    # print(f"Biomass R2: {get_r2(df, 'Biomass', 'Biomass simulated')}")

    nrmse_carotene = get_rmse_v2(df, 'Carotene concentration', 'Carotene concentration simulated') / df['Carotene concentration'].mean()
    nrmse_lutein = get_rmse_v2(df, 'Lutein concentration', 'Lutein concentration simulated') / df['Lutein concentration'].mean()
    print(f"Carotene concentration NRMSE: {nrmse_carotene}")
    print(f"Lutein concentration NRMSE: {nrmse_lutein}")
    mape_carotene = get_ae(df, 'Carotene concentration', 'Carotene concentration simulated').mean()
    mape_lutein = get_ae(df, 'Lutein concentration', 'Lutein concentration simulated').mean()
    print(f"Carotene concentration MAPE: {mape_carotene}")
    print(f"Lutein concentration MAPE: {mape_lutein}")


    plt.figure(figsize=(10, 10))
    for condition in matrix.conditions.index:
        if not condition.startswith("fachet") and not condition.startswith("Xi") and not condition.startswith("Yimei"):
            trajectory = pd.read_csv(f"{RESULTS_PATH}/concentrations/concentrations_{condition}.csv")
            plt.plot(trajectory['Phosphate_quota'], label=condition)
    plt.legend()
    plt.xlabel("Time (h)")
    plt.ylabel("Phosphate quota")
    plt.savefig(f"{RESULTS_PATH}/quotas/phosphate_quotas.png", dpi=300)

    plt.figure(figsize=(10, 10))
    for condition in matrix.conditions.index:
        if not condition.startswith("fachet") and not condition.startswith("Xi") and not condition.startswith("Yimei"):
            trajectory = pd.read_csv(f"{RESULTS_PATH}/concentrations/concentrations_{condition}.csv")
            plt.plot(trajectory['Nitrogen_quota'], label=condition)
    plt.legend()
    plt.xlabel("Time (h)")
    plt.ylabel("Nitrogen quota")
    plt.savefig(f"{RESULTS_PATH}/quotas/nitrogen_quotas.png", dpi=300)

    # plt.clf()
    # plt.figure(figsize=(10, 10))
    # for condition in matrix.conditions.index:
    #     if not condition.startswith("fachet") and not condition.startswith("Xi") and not condition.startswith("Yimei"):
    #         trajectory = pd.read_csv(f"{RESULTS_PATH}/concentrations/concentrations_{condition}.csv")
    #         plt.plot(trajectory['Chlorophyll'], label=condition)
    # plt.legend()
    # plt.xlabel("Time (h)")
    # plt.ylabel("Chlorophyll (g/gDW)")
    # plt.savefig(f"{RESULTS_PATH}/Chlorophyll_over_time.png", dpi=300)


def run_all_parallel(initial_parameters=None):
    """
    Runs all the conditions in parallel.
    Returns
    -------

    """
    if not initial_parameters:
        initial_parameters = json.load(open(f"{DATA_PATH}/parameters/initial_parameters.json", "r"))
    conditions_names = sorted(tuple(set(matrix.matrix.keys()) - {"Resume"}))
    conditions_names = set([condition for condition in conditions_names if
                            not condition.startswith("fachet") and not condition.startswith(
                                "Xi") and not condition.startswith("Yimei")])
    conditions_names = conditions_names - {"OC", "SC"}
    total_error = sum(
        progress_map(partial(evaluate_trial, initial_parameters, True), conditions_names, n_cpu=len(conditions_names),
                     process_timeout=60)) / len(conditions_names) * 100
    with open(f"{RESULTS_PATH}/logs/total_error.txt", "w") as f:
        f.write(str(round(total_error, 3)))
    generate_trials_plots()
    # print(f"Total error from set of parameters: {total_error}")


def run_condition(condition="1", initial_parameters=None):
    """
    Runs a single condition.
    Returns
    -------

    """
    if not initial_parameters:
        initial_parameters = json.load(open(f"{DATA_PATH}/parameters/initial_parameters.json", "r"))
    create_dfba_model(condition, initial_parameters, create_plots=True)


def run_all():
    """
    Runs all the conditions.
    Returns
    -------

    """
    initial_parameters = json.load(open(f"{DATA_PATH}/parameters/initial_parameters.json", "r"))

    conditions_names = tuple(set(matrix.matrix.keys()) - {"Resume"})

    with tqdm(total=len(conditions_names), desc="Running initial conditions") as pbar:
        for condition in conditions_names:
            pbar.set_description(f"Running {condition}")
            create_dfba_model(condition, initial_parameters, create_plots=True)
            pbar.update(1)


def optimize_lutein():
    parameter_optimization_ea_mo([
        "l",
        'Kaeration_lut',
        'ExA_lut',
        'a0p_lut',
        'a0_lut',
        'smoothing_factor_lut',
        'smoothing_factor_lut_p',
        'v_lut_max',
        'lutein_aeration_exponent',
    ]
    )

def optimize_carotene():
    parameter_optimization_ea([
        'l_caro',
        'Kaeration_caro',
        'ExA_caro',
        'a0',
        'a0p',
        'smoothing_factor',
        'smoothing_factor_p',
        'v_car_max',
        'carotene_aeration_exponent'
    ]
    )


def main():
    parser = argparse.ArgumentParser(description="Run specific functions from the script.")
    parser.add_argument("function", choices=["optimize_lutein", "optimize_carotene", "run_all",
                                             "run_all_parallel", "run_condition"], help="Function to run")
    parser.add_argument("--condition", type=str, help="Condition to run for run_condition function")
    args = parser.parse_args()
    initial_parameters = json.load(open(f"{RESULTS_PATH}/parameters/optimized_parameters.json", "r"))
    if args.function == "optimize_lutein":
        optimize_lutein()
    elif args.function == "optimize_carotene":
        optimize_carotene()
    elif args.function == "run_all":
        run_all()
    elif args.function == "run_all_parallel":
        run_all_parallel(initial_parameters)
    elif args.function == "run_condition":
        if args.condition:
            run_condition(args.condition, initial_parameters)
        else:
            print("Please provide a condition using --condition for the run_condition function.")
            sys.exit(1)

if __name__ == '__main__':
    main()
    # parameter_optimization_ea()
    # optimize_lutein()
    # optimize_carotene()
    # save paretto front

#     parameter_optimization_ea(
#         [
#          'Esat',
#          'KEchl',
#          'ymax',
#          'Kaeration',
#         "chl_aeration_exponent",
#          "a0chlp",
#         "smoothing_factor_chl_p",
#             "v_nitrate_max",
#             "wNmax",
#             "wNmin",
#             "VNmax",
#             "KNm"
#         ]
#     )
#     parameter_optimization_ea(
#         [
#          'K_nitrogen_quota',
#          'v_nitrate_max',
#         #  'v_polyphosphate_max',
#          'wNmax',
#          'wNmin',
#         #  'wPmin',
#          'Esat',
#          'KEchl',
#          'ymax',
#         #  'KPm',
#         #  'VPmax',
#          'Kaeration',
#          "chl_aeration_exponent",
#          "a0chlp",
#         "smoothing_factor_chl_p",
#             "VNmax",
#             "KNm"
#          ])
    # parameter_optimization_ea([
    #                         'light_conversion_factor',
    #                         "K_nitrogen_quota",
    #                         'ro0',
    #                         'ro1',
    #                         'v_nitrate_max',
    # #                         'v_polyphosphate_max',
    #                         'wNmax',
    #                         'wNmin',
    #                         # 'wPmin',
    #                         'Esat',
    # #                         # 'KEchl',
    # #                         # 'ymax',
    #                         'maximum_starch_production',
    #                         't_max',
    #                         'maximum_tag_production',
    #                         'vco2max',
    #                         'VNmax',
    # #                         'VPmax',
    #                         "KNm",
    # #                         "KPm"
    #                            ])
    # run_all()
    # initial_parameters = json.load(open(f"{RESULTS_PATH}/parameters/optimized_parameters.json", "r"))
    # run_all_parallel(initial_parameters)
    # initial_parameters = json.load(open(f"{DATA_PATH}/parameters/initial_parameters.json", "r"))
    #
    # matrix.conditions.loc["TC", "[N] mmol"] = 3
    # run_condition("OC", initial_parameters)
    # run_condition("SC", initial_parameters)

    # tc = evaluate_trial_mo(initial_parameters, condition="OC")
    # sc = evaluate_trial_mo(initial_parameters, condition="SC")
    # print(f"OC: {tc}\nSC: {sc}")
    # plt.show()
    # from results_analysis import plot_caros
    # plot_caros()
    # pass
