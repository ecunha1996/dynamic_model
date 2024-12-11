def get_dynamic_expressions(parameters):
    drhs = {
            "ActiveBiomass": parameters["mu"] * parameters["X"],
            "Biomass": parameters["total_growth_rate"] * parameters["X"],
            "Nitrogen_quota": -parameters["v_N"] - parameters["total_growth_rate"] * parameters["n_quota"],
            "Phosphate_quota": -parameters["v_P"] - parameters["total_growth_rate"] * parameters["p_quota"],
            "Light": 0,
            "Nitrate": parameters["v_N"] * parameters["X"],
            "Phosphate": parameters["v_P"] * parameters["X"],
            "Starch": parameters["starch_production"] - parameters["total_growth_rate"] * parameters["starch"],
            "Starch_concentration": parameters["starch_production"] * parameters["X"],
            "TAG": parameters["tag_production"] - parameters["total_growth_rate"] * parameters["tag"],
            # "Glycerol": parameters["glycerol_production"] - parameters["total_growth_rate"] * parameters["glycerol"],
            "Carotene": parameters["caro_production"] - parameters["total_growth_rate"] * parameters["carotene"],
            "Lutein": parameters["lutein_production"] - parameters["total_growth_rate"] * parameters["lutein"],
            "Chlorophyll": parameters["chl_production"] - parameters["total_growth_rate"] * parameters["chlorophyll"]
            }
    return drhs
