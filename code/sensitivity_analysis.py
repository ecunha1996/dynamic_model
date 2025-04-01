import sys

import matplotlib.pyplot as plt
from SALib.sample import morris
from SALib.analyze import morris as morris_analysis
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

sns.set_theme(context='paper', palette="colorblind",  font='Arial')
plt.rcParams['axes.labelsize'] = 7
plt.rcParams['axes.titlesize'] = 8
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": "Arial",
})
plt.rcParams['axes.labelsize'] = 7
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7
plt.rcParams['legend.fontsize'] = 7
plt.rcParams['font.size'] = 7

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

parameter_mapping = {
    "v_polyphosphate_max": r"$\nu_{PO_4,intra,max}$",
    "VPmax": r"$\nu_{HPO_4,max}$",
    "KPm": r"$Km_{HPO_4}$",

    "VNmax": r"$\nu_{NO_3,max}$",
    "KNm": r"$Km_{NO_3}$",

    "v_nitrate_max": r"$\nu_{NO_3,intra,max}$",

    "wNmax": r"$\omega_{N_{max}}$",
    "wNmin": r"$\omega_{N_{min}}$",
    "wPmin": r"$\omega_{P_{min}}$",

    "K_nitrogen_quota": r"$q_m$",

    "light_conversion_factor": r"$x_{dim}$",

    "ro0": r"$\eta_0$",
    "ro1": r"$\eta_1$",

    "Esat": r"$E_{sat}$",
    "KEchl": r"$K_E$",
    "ymax": r"$\gamma_{max}$",

    "Kaeration": r"$K_{A,chl}$",
    "chl_aeration_exponent": r"$n_{A,chlo}$",

    "a0chlp": r"$\alpha_{a_{0,chl}}$",
    "smoothing_factor_chl_p": r"$\eta_{chl,P}$",

    "maximum_starch_production": r"$\nu_{starch,max}$",
    "t_max": r"$t_{max}$",
    "vco2max": r"$\nu_{CO_2,max}$",

    "maximum_tag_production": r"$\nu_{TAG,max}$",

    "v_car_max": r"$\nu_{car,max}$",
    "v_lut_max": r"$\nu_{lut,max}$",

    "ExA_caro": r"$E_{xA,caro}$",
    "ExA_lut": r"$E_{xA,lut}$",

    "l_caro": r"$m_{caro}$",
    "l": r"$m_{lut}$",

    "Kaeration_caro": r"$K_{A,caro}$",
    "Kaeration_lut": r"$K_{A,lut}$",

    "carotene_aeration_exponent": r"$n_{A,caro}$",
    "lutein_aeration_exponent": r"$n_{A,lut}$",

    "a1": r"$\alpha_{a_{1,N,caro}}$",
    "a1_lut": r"$\alpha_{a_{1,N,lut}}$",

    "a0p": r"$\alpha_{a_{0,P,caro}}$",
    "a0p_lut": r"$\alpha_{a_{0,P,lut}}$",

    "a0": r"$\alpha_{a_{0,N,caro}}$",
    "a0_lut": r"$\alpha_{a_{0,N,lut}}$",

    "smoothing_factor": r"$\eta_{caro,N}$",
    "smoothing_factor_lut": r"$\eta_{lut,N}$",

    "smoothing_factor_p": r"$\eta_{caro,P}$",
    "smoothing_factor_lut_p": r"$\eta_{lut,P}$"
}


def evaluate_nutrient(individual):
    for ind in individual:
        matrix.conditions.loc["OC", ind[0]] = round(ind[1], 2)
    initial_parameters = json.load(open(f"{RESULTS_PATH}/parameters/optimized_parameters.json", "r"))
    concentrations_tc, trajectories_tc = create_dfba_model("OC", initial_parameters, False)
    return concentrations_tc["Carotene_concentration"].iloc[-1]*1000, concentrations_tc["Lutein_concentration"].iloc[-1]*1000, concentrations_tc["ActiveBiomass"].iloc[-1]


def main():
    parameters_to_evaluate = ("[P] mmol", "Light (umol/m^2.s)", "Aeration rate")

    nitrate_range = np.arange(1, 15.5, 0.5)
    phosphate_range = np.arange(0.05, 0.4, 0.05)
    light_range = np.arange(10, 1000, 10)
    aeration_range = np.arange(400, 1000, 50)
    res = {}
    # for i in nitrate_range:
    #     res[i] = evaluate_nutrient(["[N] mmol", i])
    # sns.lineplot(x=res.keys(), y=res.values())
    # plt.show()

    use_cache = True
    if not use_cache:
        carotene, lutein = [], []
        biomass = []
        for i in phosphate_range:
            for j in nitrate_range:
                res = evaluate_nutrient([("[P] mmol", i), ("[N] mmol", j)])
                carotene.append([i, j, res[0]])
                lutein.append([i, j, res[1]])
                biomass.append([i, j, res[2]])

        # join the biomass, carotene, and lutein  results in a single df and save it as a tsv
        df = pd.DataFrame(biomass, columns=["Phosphate", "Nitrate", "Biomass"])
        df["Carotene"] = [i[2] for i in carotene]
        df["Lutein"] = [i[2] for i in lutein]
        df.to_csv(f"{RESULTS_PATH}/sensitivity_analysis.tsv", sep="\t", index=False)

    else:
        df = pd.read_csv(f"{RESULTS_PATH}/sensitivity_analysis.tsv", sep="\t")
        biomass = df[["Phosphate", "Nitrate", "Biomass"]].values
        carotene = df[["Phosphate", "Nitrate", "Carotene"]].values
        lutein = df[["Phosphate", "Nitrate", "Lutein"]].values

    # round nitrate and phosphate range
    nitrate_range = np.round(nitrate_range, 2)
    phosphate_range = np.round(phosphate_range, 2)

    heatmap_data_carotene = np.zeros((len(phosphate_range), len(nitrate_range)))
    heatmap_data_lutein = np.zeros((len(phosphate_range), len(nitrate_range)))
    for p_idx, p in enumerate(phosphate_range):
        for n_idx, n in enumerate(nitrate_range):
            heatmap_data_carotene[p_idx, n_idx] = carotene[p_idx * len(nitrate_range) + n_idx][2]
            heatmap_data_lutein[p_idx, n_idx] = lutein[p_idx * len(nitrate_range) + n_idx][2]


    # plt.figure(figsize=(7.08, 3))
    # sns.heatmap(heatmap_data_carotene, xticklabels=nitrate_range, yticklabels=phosphate_range, cmap="viridis")
    # plt.xlabel("Initial Nitrate Concentration (mM)")
    # plt.ylabel("Initial Phosphate Concentration (mM)")
    # plt.title("Carotene Concentration Heatmap")
    # plt.show()
    #
    # # Plot the heatmap
    # plt.figure(figsize=(7.08, 3))
    # sns.heatmap(heatmap_data_lutein, xticklabels=nitrate_range, yticklabels=phosphate_range, cmap="viridis")
    # plt.xlabel("Initial Nitrate Concentration (mM)")
    # plt.ylabel("Initial Phosphate Concentration (mM)")
    # plt.title("Lutein Concentration Heatmap")
    # plt.show()

    fig, axs = plt.subplots(2, 1, figsize=(7.08, 4.7))

    heatmap_data_carotene = heatmap_data_carotene[::-1]
    heatmap_data_lutein = heatmap_data_lutein[::-1]
    phosphate_range = phosphate_range[::-1]

    sns.heatmap(heatmap_data_carotene, xticklabels=nitrate_range, yticklabels=phosphate_range, cmap="viridis", ax=axs[0])
    axs[0].set_xlabel("Initial Nitrate Concentration (mM)")
    axs[0].set_ylabel("Initial Phosphate Concentration (mM)")
    cbar1 = axs[0].collections[0].colorbar
    cbar1.set_label(r"$\beta$-Carotene Concentration (mg/L)")
    # axs[0].set_title("Carotene Concentration (mg/L)")
    axs[0].text(-0.05, 1.05, "A", horizontalalignment='center', verticalalignment='center', transform=axs[0].transAxes, fontsize=10, fontweight='bold')

    axs[0].annotate(f"OC", (13.1, 3.7), color="white", fontweight='bold')
    axs[0].annotate(f"SC", (18, 5.7), color="white", fontweight='bold')

    sns.heatmap(heatmap_data_lutein, xticklabels=nitrate_range, yticklabels=phosphate_range, cmap="viridis", ax=axs[1])
    axs[1].set_xlabel("Initial Nitrate Concentration (mM)")
    axs[1].set_ylabel("Initial Phosphate Concentration (mM)")
    cbar2 = axs[1].collections[0].colorbar
    cbar2.set_label("Lutein Concentration (mg/L)")
    axs[1].text(-0.05, 1.05, "B", horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes, fontsize=10, fontweight='bold')
    # axs[1].set_title("Lutein Concentration (mg/L)")

    axs[1].annotate(f"OC", (13.1, 3.7), color="white", fontweight='bold')
    axs[1].annotate(f"SC", (18, 5.7), color="white", fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{RESULTS_PATH}/carotenoid_production.pdf", bbox_inches="tight", format="pdf", dpi=600)
    plt.show()

    heatmap_data = np.zeros((len(phosphate_range), len(nitrate_range)))
    for p_idx, p in enumerate(phosphate_range):
        for n_idx, n in enumerate(nitrate_range):
            heatmap_data[p_idx, n_idx] = biomass[p_idx*len(nitrate_range) + n_idx][2]

    heatmap_data = heatmap_data[::-1]
    plt.figure(figsize=(7.08, 3))
    sns.heatmap(heatmap_data, xticklabels=nitrate_range, yticklabels=phosphate_range, cmap="viridis")
    plt.xlabel("Initial Nitrate Concentration (mM)")
    plt.ylabel("Initial Phosphate Concentration (mM)")
    plt.title("Biomass")

    plt.tight_layout()
    plt.savefig(f"{RESULTS_PATH}/biomass_production.pdf", bbox_inches="tight", format="pdf", dpi=600)


    plt.show()

def fitness_func(initial_parameters, conditions_names, parameters_names, target=None):
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
    try:
        parameters = {}
        for i, parameter_name in enumerate(parameters_names):
            parameters[parameter_name] = initial_parameters[i]
        total_error = round(sum(Parallel(n_jobs=len(conditions_names), timeout=30, backend="multiprocessing")(
            delayed(evaluate_trial)(parameters, condition=condition, targets=[target]) for condition in
            conditions_names)), 3)
        if total_error >= 1e3:
            total_error = np.inf
    except Exception as e:
        print(e)
        total_error =  np.inf
    return total_error

def sensitivity_analysis(target):
    initial_parameters = json.load(open(f"{RESULTS_PATH}/parameters/optimized_parameters.json", "r"))
    parameter_bounds = json.load(open(f"../data/parameters/parameters_bounds.json", "r"))
    parameter_bounds = {k: v for k, v in parameter_bounds.items() if k in initial_parameters}
    parameter_names = list(parameter_bounds.keys())
    parameter_values = list(parameter_bounds.values())

    problem = {
        'num_vars': len(parameter_values),
        'names': parameter_names,
        'bounds': parameter_values
    }
    number_of_trajectories = 500
    param_values = morris.sample(problem, N=number_of_trajectories, num_levels=4)
    # with open(f"{RESULTS_PATH}/param_values.json", "w") as f:
    #     json.dump(param_values.tolist(), f)

    Y = np.array([fitness_func(params, ["SC", "OC"], parameter_names, target=target) for params in param_values])


    # valid_indices = ~np.isnan(Y)
    # param_values_filtered = param_values[valid_indices]
    # Y_filtered = Y[valid_indices]

    # param_values = np.array(json.load(open(f"{RESULTS_PATH}/param_values.json", "r")))
    # Y =  np.array(json.load(open(r"/home/ecunha/dynamic_model/results/Y_Biomass.json")))

    # valid_indices = ~np.isinf(Y)
    # param_values_filtered = param_values[valid_indices]
    # Y_filtered = Y[valid_indices]

    with open(f"{RESULTS_PATH}/sensitivity/Y_{target}.json", "w") as f:
        json.dump(Y.tolist(), f)

    # with open(f"{RESULTS_PATH}/Y_filtered_{target}.json", "w") as f:
    #     json.dump(Y_filtered.tolist(), f)

    D = len(parameter_names)  # Number of parameters
    N = number_of_trajectories  # Original number of trajectories

    # Reshape Y into (N, D+1) so each row represents a trajectory
    Y_reshaped = Y.reshape(N, D + 1)

    # Identify valid trajectories (only keep those without NaNs or Infs)
    valid_trajectories_mask = np.all(np.isfinite(Y_reshaped), axis=1)

    # Count the number of valid trajectories
    max_valid_trajectories = np.sum(valid_trajectories_mask)

    print(f"Maximum number of valid trajectories: {max_valid_trajectories}")

    # Filter param_values accordingly to keep only valid trajectories
    param_values_filtered = param_values.reshape(N, D + 1, D)[valid_trajectories_mask].reshape(-1, D)
    Y_filtered = Y_reshaped[valid_trajectories_mask].flatten()

    Si = morris_analysis.analyze(problem, param_values_filtered, Y_filtered, conf_level=0.95, print_to_console=True, seed=42)

    Si_serializable = {key: np.ma.filled(value, np.nan).tolist() if isinstance(value, np.ma.MaskedArray) else value.tolist() if isinstance(value, np.ndarray) else value for key, value in Si.items()}

    with open(f"{RESULTS_PATH}/sensitivity/sensitivity_analysis_{target}.json", "w") as f:
        json.dump(Si_serializable, f)

    return Si


def plot_s1(Si, target):
    # fig, ax = plt.subplots(1, 1, figsize=(7.08, 10))
    # sns.barplot(x=Si["mu_star"], y=Si['names'], ax=ax)
    # ax.set_xlabel("Sensitivity Index")
    # ax.set_ylabel("Parameter")
    # plt.tight_layout()
    # plt.savefig(f"{RESULTS_PATH}/sensitivity_analysis_{target}.pdf", bbox_inches="tight", format="pdf", dpi=600)
    # plt.show()

    threshold = 0.05

    df = pd.DataFrame({
        "Parameter": Si["names"],
        "Mu* (Mean Abs Effect)": Si["mu_star"],
        "Sigma (Std Dev)": Si["sigma"],
        "Confidence Interval": Si["mu_star_conf"]
    })

    df_sorted = df.sort_values(by="Mu* (Mean Abs Effect)", ascending=False)
    df_sorted["Parameter"] = df_sorted["Parameter"].map(parameter_mapping)
    # Plot
    plt.figure(figsize=(10, 10))
    sns.barplot(x=df_sorted["Mu* (Mean Abs Effect)"], y=df_sorted["Parameter"], alpha=0.7)
    plt.ylabel("Parameter")
    plt.xlabel("Mu* (Mean Abs Effect)")
    plt.title("Morris Sensitivity Analysis (Mu*)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_PATH}/sensitivity/sensitivity_analysis_mu_{target}.pdf", bbox_inches="tight", format="pdf", dpi=600)
    plt.show()

    plt.figure(figsize=(7.08, 5))
    plt.scatter(df["Mu* (Mean Abs Effect)"], df["Sigma (Std Dev)"], c='b', alpha=0.6)
    plt.xlabel("Mu* (Mean Abs Effect)")
    plt.ylabel("Sigma (Std Dev)")
    plt.title("Parameter Influence: Mu* vs Sigma")
    plt.axhline(y=threshold, color='r', linestyle='--', label="Threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_PATH}/sensitivity/sensitivity_analysis_mu_sigma_{target}.pdf", bbox_inches="tight", format="pdf", dpi=600)
    plt.show()



if __name__ == "__main__":
    # main()
    parameter_bounds = json.load(open(f"../data/parameters/parameters_bounds.json", "r"))
    parameter_names = list(parameter_bounds.keys())
    targets = ["Biomass", "Carotene", "Lutein"] #
    for target in targets:
        si = sensitivity_analysis(target)
        plot_s1(si, target)
