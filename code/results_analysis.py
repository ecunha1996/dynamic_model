from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gsmmutils.experimental.exp_matrix import ExpMatrix
from gsmmutils.graphics import plot_concentrations
import seaborn as sns

sns.set_theme(context='paper', style='ticks', palette="colorblind",  font='Arial')
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

def plot_caros():
    sc_conc = pd.read_csv("/home/ecunha/dynamic_model/results/concentrations/concentrations_SC.csv")
    car_exp = {4: 0.00045116, 8: 0.000333513, 10: 0.00047447, 12: 0.000371476, 14: 0.000161355}
    lut_exp = {4: 0.002943876
        , 8: 0.001903962
        , 10: 0.002526625
        , 12: 0.001929135
        , 14: 0.001478111
               }
    chl = {4: 0.005994556, 8: 0.005155043, 10: 0.00599604, 12: 0.004537865, 14: 0.003939947}

    axs = sc_conc.plot(x="time", y="Chlorophyll", title="Chlorophyll SC")
    axs.scatter(list(chl.keys()), list(chl.values()))
    axs.set_ylabel("Chlorophyll (g/gDW)")
    axs.set_xlabel("Time (d)")
    plt.show()

    axs = sc_conc.plot(x="time", y="Carotene", title="Carotene SC")
    axs.scatter(list(car_exp.keys()), list(car_exp.values()))
    axs.set_ylabel("Carotene (g/gDW)")
    axs.set_xlabel("Time (d)")
    plt.show()

    axs = sc_conc.plot(x="time", y="Lutein", title="Lutein SC")
    axs.scatter(list(lut_exp.keys()), list(lut_exp.values()))
    axs.set_ylabel("Lutein (g/gDW)")
    axs.set_xlabel("Time (d)")
    plt.show()

    car_exp = {4: 0.000166591, 8: 0.000267999
        , 10: 0.000462569
        , 12: 0.000374841
        , 14: 0.000165156}
    lut_exp = {4: 0.001087013
        , 8: 0.001529954
        , 10: 0.002463251
        , 12: 0.001946606
        , 14: 0.001512929
               }

    axs = sc_conc.plot(x="time", y="Carotene_concentration", title="Carotene Concentration SC")
    axs.scatter(list(car_exp.keys()), list(car_exp.values()))
    plt.show()

    axs = sc_conc.plot(x="time", y="Lutein_concentration", title="Lutein Concentration SC")
    axs.scatter(list(lut_exp.keys()), list(lut_exp.values()))
    plt.show()

    axs = sc_conc.plot(x="time", y="Nitrogen_quota", title="Nitrogen Quota SC")
    axs.set_ylabel("Nitrogen Quota (mmol N/gDW)")
    axs.set_xlabel("Time (d)")
    plt.show()

    axs = sc_conc.plot(x="time", y="Phosphate_quota", title="Phosphate Quota SC")
    axs.set_ylabel("Phosphorus Quota (mmol P/gDW)")
    axs.set_xlabel("Time (d)")
    plt.show()

    sc_conc = pd.read_csv("/home/ecunha/dynamic_model/results/concentrations/concentrations_TC.csv")

    chl = {4: 0.006979269, 8: 0.011705649, 12: 0.0112052, 16: 0.008487263}

    car_exp = {4: 0.000586728,
               8: 0.000758562,
               12: 0.000829533,
               16: 0.000680184
               }
    lut_exp = {4: 0.003536932, 8: 0.004008158,
               12: 0.004144134, 16: 0.003374838}

    axs = sc_conc.plot(x="time", y="Chlorophyll", title="Chlorophyll OC")
    axs.scatter(list(chl.keys()), list(chl.values()))
    axs.set_ylabel("Chlorophyll (g/gDW)")
    axs.set_xlabel("Time (d)")
    plt.show()

    axs = sc_conc.plot(x="time", y="Carotene", title="Carotene OC")
    axs.scatter(list(car_exp.keys()), list(car_exp.values()))
    axs.set_ylabel("Carotene (g/gDW)")
    axs.set_xlabel("Time (d)")
    plt.show()

    axs = sc_conc.plot(x="time", y="Lutein", title="Lutein OC")
    axs.scatter(list(lut_exp.keys()), list(lut_exp.values()))
    axs.set_ylabel("Lutein (g/gDW)")
    axs.set_xlabel("Time (d)")
    plt.show()

    car_exp = {4: 0.000247949,
               8: 0.000676543,
               12: 0.001263697,
               16: 0.001199714
               }

    lut_exp = {4: 0.001494693, 8: 0.003574782,
               12: 0.006313106, 16: 0.005952564
               }

    axs = sc_conc.plot(x="time", y="Carotene_concentration", title="Carotene Concentration OC")
    axs.scatter(list(car_exp.keys()), list(car_exp.values()))
    plt.show()

    axs = sc_conc.plot(x="time", y="Lutein_concentration", title="Lutein Concentration OC")
    axs.scatter(list(lut_exp.keys()), list(lut_exp.values()))
    plt.show()

    axs = sc_conc.plot(x="time", y="Nitrogen_quota", title="Nitrogen Quota OC")
    axs.set_ylabel("Nitrogen Quota (mmol N/gDW)")
    axs.set_xlabel("Time (d)")
    plt.show()

    axs = sc_conc.plot(x="time", y="Phosphate_quota", title="Phosphate Quota OC")
    axs.set_ylabel("Phosphorus Quota (mmol P/gDW)")
    axs.set_xlabel("Time (d)")
    plt.show()

def make_plot(condition, conc, pigment, exp, std, axs):
    conc[pigment] = conc[pigment]*1000
    if axs:
        sns.lineplot(x="time", y=pigment, data=conc, ax=axs)
    else:
        axs = conc.plot(x="time", y=pigment, title=f"{pigment} {condition}")
    # axs.scatter(list(exp.keys()), list(exp.values()))
    axs.errorbar(list(exp.keys()), [value for key, value in exp.items() if key in std.keys() ], yerr=list(std.values()), fmt='o', color='black', ecolor='black', elinewidth=1, capsize=3)
    axs.set_ylabel(f"{pigment} (mg/gDW)")
    axs.set_xlabel("Time (d)")

def plot_caro(matrix, pigment="Carotene", axs=None):
    sc_conc = pd.read_csv("/home/ecunha/dynamic_model/results/concentrations/concentrations_SC.csv")
    tc_conc = pd.read_csv("/home/ecunha/dynamic_model/results/concentrations/concentrations_TC.csv")
    conditions = {'SC': sc_conc, "TC": tc_conc}
    pigments_cols_map = {"Carotene": ("Caro", "Caro_sd"), "Lutein": ("Lutein", "Lutein_sd")}
    for j, (condition, concentrations) in enumerate(conditions.items()):
        car_exp = OrderedDict(matrix.matrix[condition][pigments_cols_map[pigment][0]].dropna().to_dict())
        car_exp = {int(key): value*1000 for key, value in car_exp.items() if int(key)>0}
        caro_std = OrderedDict(matrix.matrix[condition][pigments_cols_map[pigment][1]].dropna().to_dict())
        caro_std = OrderedDict({int(key): value*1000 for key, value in caro_std.items() if int(key)>0})
        make_plot(condition, concentrations, pigment, car_exp, caro_std, axs[j])


def plot_1(matrix):
    fig, axs = plt.subplots(4, 2, figsize=(7.08, 6))
    plt.subplots_adjust(hspace=0.40, wspace=0.30)
    sc_conc = pd.read_csv("/home/ecunha/dynamic_model/results/concentrations/concentrations_SC.csv")
    tc_conc = pd.read_csv("/home/ecunha/dynamic_model/results/concentrations/concentrations_TC.csv")
    conditions = OrderedDict({'SC': sc_conc, "TC": tc_conc})
    for i, (condition, concentrations) in enumerate(conditions.items()):
        indexes = matrix.matrix[condition].index.astype(float)
        indexes = [e for e in indexes]
        experimental = [(indexes, matrix.matrix[condition]["DW"].tolist())]
        sns.lineplot(x="time", y="Biomass", data=concentrations, ax=axs[0, i], label = "Biomass")
        sns.lineplot(x="time", y="ActiveBiomass", data=concentrations, ax=axs[0, i], label = "Active Biomass")
        for index, exp in enumerate(experimental):
            sns.scatterplot(x=exp[0], y=exp[1], ax=axs[0, i], label=f"Experimental")
        if i==0: axs[0, i].set_ylabel("Biomass (g/L)")
        else: axs[0, i].set_ylabel("")
        axs[0, i].set_xlabel("")
        axs[0, i].legend(handlelength=1, handleheight=1)
        tmp = sns.lineplot(x="time", y="Nitrate", data=concentrations, ax=axs[1, i], label="Nitrate")
        tmp.set_yticks(np.arange(0, concentrations.Nitrate.max()+1, 2))
        if i == 0:
            tmp.set_ylabel("Nitrate (g/L)")
        else:
            tmp.set_ylabel("")
        twinx = tmp.twinx()
        sns.lineplot(x="time", y="Phosphate", data=concentrations, ax=twinx, label="Phosphate", color='orange')
        twinx.set_yticks(np.arange(0, concentrations.Phosphate.max()+0.01, 0.05))
        twinx.set_yticklabels(np.round(np.arange(0, concentrations.Phosphate.max()+0.01, 0.05), 2), fontsize=7)
        tmp.legend_.remove()
        if i == 1:
            twinx.set_ylabel("Phosphate (g/L)")
        else:
            twinx.set_ylabel("")
        lines, labels = axs[1, i].get_legend_handles_labels()
        lines2, labels2 = twinx.get_legend_handles_labels()
        twinx.legend(lines + lines2, labels + labels2, loc='upper right', handlelength=1, handleheight=1)
        axs[1, i].set_xlabel("")
    plot_caro(matrix, "Carotene", axs[2])
    plot_caro(matrix, "Lutein", axs[3])
    axs[2, 0].set_xlabel("")
    axs[2, 1].set_xlabel("")
    axs[2][1].set_ylabel("")
    axs[3][1].set_ylabel("")
    axs[3, 0].set_xlabel("Time (d)")
    axs[3, 0].set_xlabel("Time (d)")
    counter = 0
    for i, axis_set in enumerate(axs):
        for j, axis in enumerate(axis_set):
            # axis.spines['top'].set_visible(False)
            # axis.spines['right'].set_visible(False)
            # set x_ticklables until the max +1
            axis.set_xticks(np.arange(0, max(axis.get_xticks())-1.5, 2))
            axis.text(-0.1, 1.1, ALPHABET[counter], transform=axis.transAxes, weight='bold', fontsize=10)
            counter += 1
    # plt.tight_layout()
    plt.savefig(f"{RESULTS_PATH}/biomass_nutrient_carotenoids.pdf", dpi=600, bbox_inches='tight', format='pdf')
    plt.show()

def plot_2(matrix):
    fig, axs = plt.subplots(2, 2, figsize=(7.08, 4))
    plt.subplots_adjust(hspace=0.40, wspace=0.30)
    sc_conc = pd.read_csv("/home/ecunha/dynamic_model/results/concentrations/concentrations_SC.csv")
    tc_conc = pd.read_csv("/home/ecunha/dynamic_model/results/concentrations/concentrations_TC.csv")
    conditions = OrderedDict({'SC': sc_conc, "TC": tc_conc})
    for i, (condition, concentrations) in enumerate(conditions.items()):
        sns.lineplot(x="time", y="Nitrogen_quota", data=concentrations, ax=axs[0, i])
        sns.lineplot(x="time", y="Phosphate_quota", data=concentrations, ax=axs[1, i])
    axs[0, 0].set_ylabel("Nitrogen Quota (mmol N/gDW)")
    axs[0, 1].set_ylabel("")
    axs[1, 0].set_ylabel("Phosphate Quota (mmol P/gDW)")
    axs[1, 1].set_ylabel("")
    counter = 0
    for i, axis_set in enumerate(axs):
        for j, axis in enumerate(axis_set):
            axis.set_xticks(np.arange(0, max(axis.get_xticks())-1.5, 2))
            axis.text(-0.1, 1.1, ALPHABET[counter], transform=axis.transAxes, weight='bold', fontsize=10)
            counter += 1
    plt.savefig(f"{RESULTS_PATH}/quotas.pdf", dpi=600, bbox_inches='tight', format='pdf')
    plt.show()

if __name__ == "__main__":
    DATA_PATH = "../data"
    RESULTS_PATH = f"../results"
    matrix = ExpMatrix(f"{DATA_PATH}/experimental/Matriz- DCCR Dunaliella salina_dfba_new.xlsx", conditions="Resume")
    # plot_caro(matrix)
    plot_1(matrix)
    plot_2(matrix)
