from .units import ureg

Co = ureg('1920.0 J/kg/K')     # heat capacity of organic component of soil, J/kg/K
# Specific heat of dry air at constant pressure [J/kg/K]. This is the *intercept of
# the temperature-dependence fit* used by ``micrometeorology.cp_d``, cp_d(T) = Cpd +
# (T[degC] + 23.12)^2 / 3364, and not a free choice of reference value: the quadratic
# term is a correction to this particular base, so the two have to come from the same
# parameterisation. A reference value taken from elsewhere leaves cp_d low and
# the sensible heat flux with it.
Cpd = ureg('1005 J/kg/K')
g = ureg('9.81 m/s^2')           # gravitation constant
k = 0.4        # von Karmans constant
Mco2 = ureg('0.04401 kg/mol')  # molecular weight of carbon dioxide, kg/mol
Md = ureg('0.02897 kg/mol')   # molecular weight of dry air, kg/mol
Mv = ureg('0.01802 kg/mol') # 0.01802   # molecular weight of water vapour, kg/mol
Mp = ureg('0.00100728 kg/mol')  # proton mass; a PTR-MS species is detected as [M+H]+, so its neutral mass is m/z - Mp
mu = Md/Mv     # ratio of dry air molecular weight to water vapour molecular weight
R = ureg('8.314462618 J/mol/K')  # universal gas constant, CODATA 2018 (exact since the 2019 SI); Rd and Rv below are EddyPro's rounded specific constants, not R/Md and R/Mv
Rd = ureg('287.04 J/kg/K')    # gas constant for dry air, J/kg/K
Rv = ureg('461.5 J/kg/K')     # gas constant for water vapour, J/kg/K
P0a = ureg('10**5 Pa')  # reference pressure for potential temperature, Pa
# dictionary of instrument characteristics
# used in pfp_ts.MassmanStandard(), pfp_compliance.l1_check_sonic_type() and
# pfp_compliance.l1_check_irga_type()
# 'no_sonic' is for IRGAs used in profile measurements
instruments = {"sonics": {"CSAT3": {"lwVert": 0.115, "lwHor": 0.058, "lTv": 0.115},
                          "CSAT3A": {"lwVert": 0.115, "lwHor": 0.058, "lTv": 0.115},
                          "CSAT3B": {"lwVert": 0.115, "lwHor": 0.058, "lTv": 0.115},
                          "WindMaster-Pro": {"lwVert": 0.106, "lwHor": 0.107, "lTv": 0.106}},
               "irgas": {"open_path": {"Li-7500": {"dIRGA": 0.0095, "lIRGA": 0.127},
                                       "Li-7500A": {"dIRGA": 0.0095, "lIRGA": 0.127},
                                       "Li-7500RS": {"dIRGA": 0.0095, "lIRGA": 0.127},
                                       "Li-7500DS": {"dIRGA": 0.0095, "lIRGA": 0.127},
                                       "EC150": {"dIRGA": 0.01, "lIRGA": 0.154},
                                       "IRGASON": {"dIRGA": 0.01, "lIRGA": 0.154}},
                         "closed_path": {"Li-7200": {"dIRGA": 0.0064, "lIRGA": 0.125},
                                         "Li-7200RS": {"dIRGA": 0.0064, "lIRGA": 0.125},
                                         "Li-7200DS": {"dIRGA": 0.0064, "lIRGA": 0.125},
                                         "EC155": {"dIRGA": 0.008, "lIRGA": 0.120},
                                         "Li-830": {"dIRGA": None, "lIRGA": None},
                                         "Li-840": {"dIRGA": None, "lIRGA": None},
                                         "Li-850": {"dIRGA": None, "lIRGA": None},
                                         "None": {"dIRGA": None, "lIRGA": None}}}}