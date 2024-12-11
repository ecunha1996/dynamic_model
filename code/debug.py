from collections import OrderedDict

import matplotlib.pyplot as plt
import pandas as pd
from gsmmutils.experimental.exp_matrix import ExpMatrix



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

def make_plot(condition, conc, exp, std):
    conc.Carotene = conc.Carotene*1000
    axs = conc.plot(x="time", y="Carotene", title=f"Carotene {condition}")
    # axs.scatter(list(exp.keys()), list(exp.values()))
    axs.errorbar(list(exp.keys()), [value for key, value in exp.items() if key in std.keys() ], yerr=list(std.values()), fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=1)
    axs.set_ylabel("Carotene (mg/gDW)")
    axs.set_xlabel("Time (d)")
    plt.show()

def plot_caro(matrix):
    sc_conc = pd.read_csv("/home/ecunha/dynamic_model/results/concentrations/concentrations_SC.csv")
    tc_conc = pd.read_csv("/home/ecunha/dynamic_model/results/concentrations/concentrations_TC.csv")
    conditions = {'SC': sc_conc, "TC": tc_conc}

    for condition, concentrations in conditions.items():
        car_exp = OrderedDict(matrix.matrix[condition].Caro.dropna().to_dict())
        car_exp = {int(key): value*1000 for key, value in car_exp.items()}
        caro_std = OrderedDict(matrix.matrix[condition].Caro_sd.dropna().to_dict())
        caro_std = OrderedDict({int(key): value*1000 for key, value in caro_std.items()})
        caro_std[0] =0
        caro_std.move_to_end(0, last=False)
        make_plot(condition, concentrations, car_exp, caro_std)



if __name__ == "__main__":
    DATA_PATH = "../data"
    matrix = ExpMatrix(f"{DATA_PATH}/experimental/Matriz- DCCR Dunaliella salina_dfba_new.xlsx", conditions="Resume")
    plot_caro(matrix)
