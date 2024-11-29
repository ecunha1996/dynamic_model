import os
from cobra.flux_analysis import find_blocked_reactions
from gsmmutils.experimental.exp_matrix import ExpMatrix
from gsmmutils.model.COBRAmodel import MyModel
from gsmmutils.utils.utils import get_element_in_biomass, get_molecular_weight


def create_active_biomass(model):
    biomass_copy = model.reactions.e_Biomass__cytop.copy()
    biomass_copy.id = 'e_ActiveBiomass__cytop'
    model.add_reactions([biomass_copy])
    model.objective = 'e_ActiveBiomass__cytop'
    return model


def remove_starch(model):
    # remove starch from e_carbohydrate
    # adjust other carbohydrates
    model.set_stoichiometry("e_Carbohydrate__cytop", "C00369__chlo", 0)
    model.set_stoichiometry("e_Carbohydrate__cytop", "C00052__cytop", -3.3576)
    model.set_stoichiometry("e_Carbohydrate__cytop", "C00096__cytop", -1.5818)
    model.set_stoichiometry("e_Carbohydrate__cytop", "C00935__cytop", -2.5249)
    model.set_stoichiometry("e_Carbohydrate__cytop", "C00015__cytop", 5.8826)
    model.set_stoichiometry("e_Carbohydrate__cytop", "C00035__cytop", 1.5818)
    model.set_stoichiometry("e_Carbohydrate__cytop", "C00001__cytop", 7.4644)

    # # adjust biomass
    model.set_stoichiometry("e_ActiveBiomass__cytop", "e_Carbohydrate__cytop", -0.1978)
    return model


def remove_tag(model):
    model.set_stoichiometry("e_Lipid__cytop", "C00422__lip", 0)
    model.set_stoichiometry("e_Lipid__cytop", "C00641__er", -0.1607)
    model.set_stoichiometry("e_Lipid__cytop", "C00162__cytop", -0.1307)
    model.set_stoichiometry("e_Lipid__cytop", "C13508__chlo", -0.1292)
    model.set_stoichiometry("e_Lipid__cytop", "C00344__chlo", -0.2049)
    model.set_stoichiometry("e_Lipid__cytop", "C00157__cytop", -0.0438)
    model.set_stoichiometry("e_Lipid__cytop", "C00350__cytop", -0.0489)
    model.set_stoichiometry("e_Lipid__cytop", "C01194__er", -0.0373)
    model.set_stoichiometry("e_Lipid__cytop", "C03692__chlo", -0.2690)
    model.set_stoichiometry("e_Lipid__cytop", "C06037__chlo", -0.2165)
    model.set_stoichiometry("e_Lipid__cytop", "C18169__er", -0.0978)
    model.set_stoichiometry("e_Lipid__cytop", "C05980__mito", -0.0299)
    # # adjust biomass
    model.set_stoichiometry("e_ActiveBiomass__cytop", "e_Lipid__cytop", -0.0979)
    print(model.optimize().objective_value)
    return model


def remove_glycerol(model):
    acids_copy = model.reactions.e_Acid__cytop.copy()
    acids_copy.id = 'e_Acid_no_glycerol__cytop'
    model.add_reactions([acids_copy])
    model.reactions.e_Acid__cytop.bounds = (0, 0)
    model.set_stoichiometry("e_Acid_no_glycerol__cytop", "C00116__cytop", 0)
    model.set_stoichiometry("e_Acid_no_glycerol__cytop", "C00033__cytop", -5.5505)
    model.set_stoichiometry("e_Acid_no_glycerol__cytop", "C00246__cytop", -4.4998)
    model.set_stoichiometry("e_Acid_no_glycerol__cytop", "C00163__cytop", -3.7835)
    model.set_stoichiometry("e_ActiveBiomass__cytop", "e_Acid__cytop", -0.0067)
    return model


def remove_pigments(model, remove_chl=False):
    # model.set_stoichiometry("e_Pigments_no_car_chl__cytop", "C05306__chlo", 0)
    # model.set_stoichiometry("e_Pigments_no_car_chl__cytop", "C05307__chlo", 0)
    # model.set_stoichiometry("e_Pigments_no_car_chl__cytop", "C02094__chlo", 0)
    # model.set_stoichiometry("e_Pigments_no_car_chl__cytop", "C08601__chlo", 0)
    #
    # model.set_stoichiometry("e_Pigments_no_car_chl__cytop", "C08614__chlo", -1.0633)
    # model.set_stoichiometry("e_Pigments_no_car_chl__cytop", "C05433__chlo", -0.0256)
    # model.set_stoichiometry("e_Pigments_no_car_chl__cytop", "C06098__chlo", -0.0164)
    # model.set_stoichiometry("e_Pigments_no_car_chl__cytop", "C08591__chlo", -0.1660)
    #
    # model.set_stoichiometry("e_Pigments_no_car_chl__cytop", "C08606__chlo", -0.0032)
    # model.set_stoichiometry("e_Pigments_no_car_chl__cytop", "C05432__chlo", -0.0008)
    # model.set_stoichiometry("e_Pigments_no_car_chl__cytop", "C08579__chlo", -0.0014)
    # model.set_stoichiometry("e_Pigments_no_car_chl__cytop", "C20484__chlo", -0.4527)
    #
    # model.set_stoichiometry("e_ActiveBiomass__cytop", "e_Pigment__chlo", -0.0163)

    # model.set_stoichiometry("e_Pigments_no_caro__cytop", "C05306__chlo", -0.5841)
    # model.set_stoichiometry("e_Pigments_no_caro__cytop", "C05307__chlo", -0.4167)
    # model.set_stoichiometry("e_Pigments_no_caro__cytop", "C02094__chlo", 0)
    # model.set_stoichiometry("e_Pigments_no_caro__cytop", "C20484__chlo", 0)
    # model.set_stoichiometry("e_Pigments_no_caro__cytop", "C08601__chlo", 0)
    #
    # model.set_stoichiometry("e_Pigments_no_caro__cytop", "C08614__chlo", -0.0661)
    # model.set_stoichiometry("e_Pigments_no_caro__cytop", "C06098__chlo", -0.0294)
    #
    # model.set_stoichiometry("e_Pigments_no_caro__cytop", "C08606__chlo", -0.0724)
    #
    # model.set_stoichiometry("e_ActiveBiomass__cytop", "e_Pigment__chlo", -0.0114)
    model.set_stoichiometry("e_Pigment__chlo", "C02094__chlo", 0)
    model.set_stoichiometry("e_Pigment__chlo", "C20484__chlo", 0)
    model.set_stoichiometry("e_Pigment__chlo", "C08601__chlo", 0)
    if remove_chl:
        model.set_stoichiometry("e_Pigment__chlo", "C05306__chlo", 0)
        model.set_stoichiometry("e_Pigment__chlo", "C05307__chlo", 0)
        model.set_stoichiometry("e_Pigment__chlo", "C08614__chlo", -0.4740)
        model.set_stoichiometry("e_Pigment__chlo", "C06098__chlo", -0.2105)
        model.set_stoichiometry("e_Pigment__chlo", "C08579__chlo", -0.4850)
        model.set_stoichiometry("e_Pigment__chlo", "C08606__chlo", -0.5189)
        model.set_stoichiometry("e_ActiveBiomass__cytop", "e_Pigment__chlo", -0.00159)
    else:
        model.set_stoichiometry("e_Pigment__chlo", "C05306__chlo", -0.5619)
        model.set_stoichiometry("e_Pigment__chlo", "C05307__chlo", -0.4008)
        model.set_stoichiometry("e_Pigment__chlo", "C08614__chlo", -0.0636)
        model.set_stoichiometry("e_Pigment__chlo", "C06098__chlo", -0.0283)
        model.set_stoichiometry("e_Pigment__chlo", "C08579__chlo", -0.0651)
        model.set_stoichiometry("e_Pigment__chlo", "C08606__chlo", -0.0697)
        model.set_stoichiometry("e_ActiveBiomass__cytop", "e_Pigment__chlo", -0.0119)

    return model


def correct_co2_uptake(model):
    active_biomass = model.reactions.e_ActiveBiomass__cytop
    total = abs(sum([active_biomass.metabolites[met] for met in active_biomass.reactants if active_biomass.metabolites[met] > -10]))
    print(total)
    carbon_in_biomass = get_element_in_biomass(model, "C", f"e_ActiveBiomass__cytop")
    print(carbon_in_biomass)
    # exp_matrix = pickle.load(open("experimental/Matriz- DCCR Dunaliella salina_new.pkl", "rb"))
    exp_matrix = ExpMatrix("experimental/Matriz- DCCR Dunaliella salina_new.xlsx")
    r = round(exp_matrix.get_substrate_uptake_for_trial("C", "23", exp_matrix.matrix["23"], get_molecular_weight("CO2"), get_molecular_weight("C"), carbon_in_biomass) * 24, 3)
    print(r)
    model.exchanges.EX_C00011__dra.bounds = (-r, 10000)
    print(model.optimize().objective_value)
    return model


def normalize_active_biomass(model):
    active_biomass = model.reactions.e_ActiveBiomass__cytop
    total = abs(sum([active_biomass.metabolites[met] for met in active_biomass.reactants if active_biomass.metabolites[met] > -10]))
    for met in active_biomass.reactants:
        if active_biomass.metabolites[met] > -10:
            active_biomass.metabolites[met] = model.set_stoichiometry("e_ActiveBiomass__cytop", met.id, round(active_biomass.metabolites[met] / total, 5))
    total = abs(sum([active_biomass.metabolites[met] for met in active_biomass.reactants if active_biomass.metabolites[met] > -10]))
    print(total)
    return model


def main():
    import cobra
    cobra_config = cobra.Configuration()
    cobra_config.bounds = (-10000, 10000)
    model = MyModel("models/model_ds.xml", "e_Biomass__cytop")
    model.add_medium("experimental/media.xlsx", "base_medium")
    model.setup_condition("default")
    for reaction in model.reactions:
        if reaction.lower_bound <= -500:
            reaction.lower_bound = -10000
        if reaction.upper_bound >= 500:
            reaction.upper_bound = 10000
    print("Finding blocked reactions")
    blocked = find_blocked_reactions(model, open_exchanges=True, processes=30)
    blocked = list(set(blocked) - {"PRISM_red_LED_674nm__extr", "PRISM_red_LED_array_653nm__extr"})
    print("Removing blocked reactions")
    model.remove_reactions(blocked)
    print(model.slim_optimize())
    with model:
        model = create_active_biomass(model)
        model = remove_starch(model)
        model = remove_tag(model)
        # model = remove_glycerol(model)
        model = remove_pigments(model, True)
        model = normalize_active_biomass(model)
        # model = correct_co2_uptake(model)
        model.write("models/model_dfba_no_caro.xml")


if __name__ == "__main__":
    DATA_PATH = "../data"
    os.chdir(DATA_PATH)
    main()
