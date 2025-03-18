import sys

import sympy as sp


def get_bounds(name, parameters):
    function_to_call = getattr(sys.modules[__name__], name)
    return function_to_call(parameters)


def phosphate(parameters):
    return sp.Max(0, parameters["VPmax"] * parameters["phosphorus"] / (parameters['KPm'] + parameters["phosphorus"]))


def polyphosphate(parameters):
    return sp.Max(0, parameters["v_polyphosphate_max"] * (1 - parameters["wPmin"] / parameters["p_quota"]))


def light(parameters):
    Ke = 11.5 * parameters['X'] * parameters['chlorophyll']
    Ex0 = parameters['Eo'] / (parameters['Lr'] * Ke) * (1 - sp.exp(-parameters['Lr'] * Ke))
    ro = (parameters['ro1'] * parameters['chlorophyll'] + parameters['ro0'])
    pa = (parameters['X'] * parameters['Lr'])
    Ex = ro / pa * Ex0 * parameters['light_conversion_factor']
    return Ex, Ex0


def nitrate(parameters):
    return sp.Max(sp.N(parameters["VNmax"] * parameters["nitrate"] / (parameters["KNm"] + parameters["nitrate"]) * (1 - parameters["q"])), 0)


def internal_nitrate(parameters):
    return sp.Max(0, parameters["v_nitrate_max"] * (1 - parameters['wNmin'] / parameters["n_quota"]))


def starch_consumption(parameters):
    parameters["Ks"] = 0.034
    parameters["vmax"] = 0.06 / 48660.195 * 1000
    return sp.Max(0, sp.N(parameters["vmax"] * sp.Max(0, parameters["starch_concentration"]) / (sp.Max(0, parameters["starch_concentration"]) + parameters["Ks"])))


def starch_production(parameters):
    # light_factor = parameters["Ex"] ** parameters["hill_coeff_starch"] / (parameters["Kstl"] ** parameters["hill_coeff_starch"] + parameters["Ex"] ** parameters["hill_coeff_starch"])
    return sp.Max(0, sp.N(parameters["maximum_starch_production"] * (1 - parameters["z"])))

def phi(x, rs):
    return 1 / (1 + sp.exp(-rs * x))

def carotene(parameters):
    v_car_gen = (parameters["v_car_max"] * (parameters["Ex"] ** parameters["l_caro"]) / ((parameters["ExA_caro"] ** parameters["l_caro"]) + (parameters["Ex"] ** parameters["l_caro"]))*
                 (parameters["Kaeration_caro"] ** parameters['carotene_aeration_exponent'] / (parameters["aeration"] ** parameters['carotene_aeration_exponent'] +
                                                                                         parameters["Kaeration_caro"] ** parameters['carotene_aeration_exponent']))
                 )
    vcar = (v_car_gen * phi(parameters["a1"] * parameters["Ex"] + parameters["a0"] -  parameters["nitrogen_mass_quota"], parameters["smoothing_factor"])
            # *phi(parameters["a1"] * parameters["Ex"] - parameters["a0p"] + parameters["phosphate_mass_quota"], parameters["smoothing_factor_p"]))
            * phi(-parameters["a0p"] + parameters["phosphate_mass_quota"], parameters["smoothing_factor_p"]))
    return sp.Max(0, sp.N(vcar))


def lutein(parameters):
    v_lut_gen = (parameters["v_lut_max"] * (parameters["Ex"] ** parameters["l"]) / ((parameters["ExA_lut"] ** parameters["l"]) + (parameters["Ex"] ** parameters["l"]))*
                 (parameters["Kaeration_lut"] ** parameters['lutein_aeration_exponent'] / (parameters["aeration"] ** parameters['lutein_aeration_exponent'] +
                                                                                           parameters["Kaeration_lut"] ** parameters['lutein_aeration_exponent'])))
    vlut = (v_lut_gen * phi(parameters["a1_lut"] * parameters["Ex"] + parameters["a0_lut"]  - parameters["nitrogen_mass_quota"], parameters["smoothing_factor_lut"]) *
            # phi(parameters["a1_lut"] * parameters["Ex"] - parameters["a0p_lut"] + parameters["phosphate_mass_quota"], parameters["smoothing_factor_lut_p"]))
            phi(-parameters["a0p_lut"] + parameters["phosphate_mass_quota"], parameters["smoothing_factor_lut_p"]))
    return sp.Max(0, sp.N(vlut))

def chlorophyll(parameters):
    def gamma(light_intensity):
        light_intensity_value = light_intensity.subs(light_intensity, parameters["Esat"])
        if  bool(light_intensity_value > parameters["Esat"]):
            light_intensity = parameters["Esat"]
        aeration_val = aeration()
        return parameters["ymax"] * (parameters["KEchl"] / (light_intensity + parameters["KEchl"])) * aeration_val

    def sum_chl(yE):
        ratio = parameters["chlorophyll"] / parameters["nitrogen_mass_quota"]
        return yE - ratio

    def aeration():
        return (parameters["Kaeration"]**parameters["chl_aeration_exponent"]) / (parameters["aeration"]**parameters["chl_aeration_exponent"] + parameters["Kaeration"]**parameters["chl_aeration_exponent"])

    gamma_val = gamma(parameters["Ex0"])
    sum_chl_val = sum_chl(gamma_val) * phi(parameters["phosphate_mass_quota"] - parameters["a0chlp"], parameters["smoothing_factor_chl_p"])
    return sum_chl_val


def tag(parameters):
    # return sp.Max(sp.N(parameters["maximum_tag_production"] / parameters["F"] / 904.78 * parameters["n"] * parameters["nacl_lipid"] * 1 / parameters["nacl"]), 0)
    # return sp.Max(sp.N(parameters["maximum_tag_production"] * parameters["n"] * parameters["nacl_lipid"]  / (parameters["nacl"] + parameters["nacl_lipid"])), 0)
    return sp.Max(sp.N(parameters["maximum_tag_production"] * parameters["n"]), 0)


def glycerol(parameters):
    max_production = (-parameters["a"] * parameters["nacl"] ** 2 + parameters["b"] * parameters["nacl"] + parameters["c"]) / parameters["X"] * (1 - parameters["glycerol"] / parameters["wgly_max"]) * (1 - parameters["p_quota"] / parameters["wPopt"])
    return sp.Max(0, sp.N(max_production))


def co2(parameters):
    return sp.Max(0, sp.N(parameters["vco2max"] * (1 - parameters["z"])))
