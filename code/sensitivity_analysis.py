import sys
sys.path.insert(0, "/opt/src")
from main import *
from results_analysis import *
from deap import base, creator, tools, algorithms
import random
import logging
import warnings
# Suppress logging for specific libraries
logging.getLogger("setuptools").setLevel(logging.CRITICAL)
logging.getLogger("distutils").setLevel(logging.CRITICAL)
logging.basicConfig(level=logging.CRITICAL)
warnings.filterwarnings("ignore", module="setuptools.*")

def evaluate_nutrient(individual):
    matrix.conditions[individual[0]].loc["TC"] = individual[1]
    initial_parameters = json.load(open(f"{RESULTS_PATH}/parameters/optimized_parameters.json", "r"))
    concentrations_tc, trajectories_tc = create_dfba_model("TC", initial_parameters, False)
    return concentrations_tc["Carotene"].iloc[-1]*1000, concentrations_tc["ActiveBiomass"].iloc[-1]


def main():
    parameters_to_evaluate = ("[P] mmol", "Light (umol/m^2.s)", "Aeration rate")

    nitrate_range = np.arange(1, 50, 1)
    phosphate_range = np.arange(0, 0.5, 0.1)
    light_range = np.arange(10, 1000, 10)
    aeration_range = np.arange(400, 1000, 50)
    res = {}
    # for i in nitrate_range:
    #     res[i] = evaluate_nutrient(["[N] mmol", i])
    # sns.lineplot(x=res.keys(), y=res.values())
    # plt.show()

    carotene = {'P': {}, 'N': {}}
    biomass = {'P': {}, 'N': {}}
    for i in phosphate_range:
        res= evaluate_nutrient(["[P] mmol", i])
        carotene['P'][i] = res[0]
        biomass['P'][i] = res[1]

    for i in nitrate_range:
        res= evaluate_nutrient(["[N] mmol", i])
        carotene['N'][i] = res[0]
        biomass['N'][i] = res[1]

    # Create a matrix for the heatmap
    heatmap_data = np.zeros((len(phosphate_range), len(nitrate_range)))
    for p_idx, p in enumerate(phosphate_range):
        for n_idx, n in enumerate(nitrate_range):
            heatmap_data[p_idx, n_idx] = carotene['P'][p] if p in carotene['P'] else 0

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, xticklabels=nitrate_range, yticklabels=phosphate_range, annot=True, cmap="YlGnBu")
    plt.xlabel("Nitrate Concentration (mmol)")
    plt.ylabel("Phosphate Concentration (mmol)")
    plt.title("Carotene Concentration Heatmap")
    plt.show()

    heatmap_data = np.zeros((len(phosphate_range), len(nitrate_range)))
    for p_idx, p in enumerate(phosphate_range):
        for n_idx, n in enumerate(nitrate_range):
            heatmap_data[p_idx, n_idx] = biomass['P'][p] if p in biomass['P'] else 0

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, xticklabels=nitrate_range, yticklabels=phosphate_range, annot=True, cmap="YlGnBu")
    plt.xlabel("Nitrate Concentration (mmol)")
    plt.ylabel("Phosphate Concentration (mmol)")
    plt.title("Biomass")
    plt.show()


if __name__ == "__main__":
    main()