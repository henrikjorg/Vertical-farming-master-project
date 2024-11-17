import numpy as np
from typing import Dict, Any
import os
import sys  # Add this import

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))  # Adjusted path

from config.utils import get_attribute

class CropModel:
    
    def __init__(self, config: Dict[str, Any]):
        # Basic configuration parameters
        self.init_X_fw_per_plant: float = get_attribute(config, 'init_X_fw_per_plant')
        self.planting_crop_density: float = get_attribute(config, 'planting_crop_density')

        # Physiological parameters
        self.initial_Xdw_to_Xsdw: float = get_attribute(config, 'initial_Xdw_to_Xsdw')
        self.c_d: float = get_attribute(config, 'c_d')
        self.leaf_to_shoot_ratio: float = get_attribute(config, 'leaf_to_shoot_ratio')
        self.leaf_diameter: float = get_attribute(config, 'leaf_diameter')

        # Growth coefficients and parameters
        self.c_tau: float = get_attribute(config, 'c_tau')
        self.c_Q10_gr: float = get_attribute(config, 'c_Q10_gr')
        self.c_gamma: float = get_attribute(config, 'c_gamma')
        self.c_gr_max: float = get_attribute(config, 'c_gr_max')
        self.c_beta: float = get_attribute(config, 'c_beta')
        self.c_alpha: float = get_attribute(config, 'c_alpha')
        self.c_resp_sht: float = get_attribute(config, 'c_resp_sht')
        self.c_resp_rt: float = get_attribute(config, 'c_resp_rt')
        self.c_Q10_resp: float = get_attribute(config, 'c_q10_resp')
        self.c_K: float = get_attribute(config, 'c_K')
        self.c_lar: float = get_attribute(config, 'c_lar')
        self.c_w: float = get_attribute(config, 'c_w')
        self.c_Gamma: float = get_attribute(config, 'c_Gamma')
        self.c_Q10_Gamma: float = get_attribute(config, 'c_Q10_Gamma')
        self.c_car_1: float = get_attribute(config, 'c_car_1')
        self.c_car_2: float = get_attribute(config, 'c_car_2')
        self.c_car_3: float = get_attribute(config, 'c_car_3')
        self.c_epsilon: float = get_attribute(config, 'c_epsilon')
        self.c_p: float = get_attribute(config, 'c_p')
        self.T_crop: float = get_attribute(config, 'T_in')
        self.CO2_in: float = get_attribute(config, 'CO2_in')

        # Climate parameters
        self.air_vel: float = get_attribute(config, 'air_vel')


        # Initialize dynamic attributes
        self.DAT: int = 0  # Days After Transplanting
        self.set_initial_attributes()

        # Initial state
        self.init_state = np.array([self.X_nsdw, self.X_sdw])


    def set_climate_model(self, climate_model):
        self.climate_model = climate_model


    def set_initial_attributes(self):
        self.init_X_fw_per_plant: float = self.init_X_fw_per_plant                              #Initial fresh weight per plant (fresh weight of 1 shoot)
        self.fresh_weight_shoot_per_plant: float = self.init_X_fw_per_plant                     #Used in env.py
        self.init_X_fw: float = self.init_X_fw_per_plant * self.planting_crop_density           #Initial fresh weight of all crop (Total value) (1 square meter)
        self.X_dw: float = self.X_dw(self.init_X_fw, self.c_d, self.c_tau)                      #Initial dry weight of all crop (Total value)
        self.dry_weight_per_plant = self.X_dw / self.planting_crop_density                      #Single plant. Used in env.py
        self.X_sdw: float = self.X_dw * self.initial_Xdw_to_Xsdw                                #Initial structural dry weight. The amount of dry weight that is structural (Total value)
        self.X_nsdw: float = self.X_dw * (1-self.initial_Xdw_to_Xsdw)                           #Initial non-structural dry weight. The amount of dry weight that is non-structural (Total value)

        self.LAI = self.LAI(self.c_lar, self.c_tau, self.X_sdw)
        self.CAC = self.CAC(self.c_K, self.LAI)
        self.f_phot: float = 0


    ####Equations from the Bae and Tiller (B&T) master thesis####

    def U_par(self, c_p, PPFD):
        # (Eq. 3.1 B&T) The photosynthetic photon flux density (PPFD) absorbed by the canopy.
        return c_p * PPFD

    def X_dw(self, X_nsdw, X_sdw):
        # (Eq. 3.2 B&T) The total dry weight of the plant.
        return X_nsdw + X_sdw

    def rate_of_X_sdw(self, r_gr, X_sdw):
        # (Eq. 3.3a B&T) The rate of change of the structural dry weight.
        return r_gr * X_sdw

    def rate_of_X_nsdw(self, c_alpha, f_phot, r_gr, X_sdw, f_resp, c_beta):
        # (Eq. 3.3b B&T) The rate of change of the non-structural dry weight.
        return c_alpha*f_phot - r_gr*X_sdw - f_resp - (1-c_beta)/c_beta * r_gr * X_sdw

    def r_gr(self, c_gr_max, X_nsdw, X_sdw, c_Q10_gr, T_crop):
        # (Eq. 3.4 B&T) r_gr: The specific growth rate
        return c_gr_max * X_nsdw / (X_sdw + X_nsdw) * c_Q10_gr**((T_crop-20) / 10)

    def f_resp(self, c_resp_sht, c_tau, X_sdw, c_resp_rt, c_Q10_resp, T_crop):
        # (Eq. 3.5 B&T) f_resp: Maintainance respiration.
        return (c_resp_sht*(1-c_tau)*X_sdw + c_resp_rt*c_tau*X_sdw) * c_Q10_resp**((T_crop-25) / 10)

    def f_phot_ODE(self, f_phot_max, CAC):
        # (Eq. 3.6 B&T) f_phot: Gross canopy photosynthesis. The photosynthetic rate of the canopy. The rate at which the canopy absorbs light and converts it into chemical energy.
        return f_phot_max * CAC

    def CAC(self, c_K, LAI):
        # (Eq. 3.7 B&T) CAC: Fraction of cultivation area cover. Fraction of the ground covered by the canopy.
        return 1 - np.exp(-c_K * LAI)

    def LAI(self, c_lar, c_tau, X_sdw):          
        # (Eq. 3.8 B&T) LAI: Leaf Area Index. Area of light-absorbing leaf surface per unit of ground surface.
        return c_lar * (1 - c_tau) * X_sdw

    def f_phot_max(self, epsilon, U_par, f_sat):          
        # (Eq. 3.9 B&T) The maximum photosynthetic rate, occurring when the canopy is fully covered.
        return (epsilon*U_par*f_sat) / (epsilon*U_par + f_sat)

    def epsilon(self, c_epsilon, CO2_in, gamma):                                    
        # (Eq. 3.10a B&T) α is the quantum yield, or light use efficiency of the absorbed light at low intensities.
        return c_epsilon * (CO2_in - gamma) / (CO2_in + 2 * gamma)

    def f_sat(self, rho_c, CO2_in, gamma, r_CO2):                                 
        # (Eq. 3.10b B&T) fsat is the light-saturated value of f_phot_max.
        return rho_c * (CO2_in - gamma) / (r_CO2)

    def gamma(self, c_gamma, c_Q_10_gamma, T_crop):                               
        # (Eq. 3.11 B&T) The CO2 compensation point.
        return c_gamma * c_Q_10_gamma ** ((T_crop - 20) / 10)

    def r_CO2(self, r_bnd, r_stm, r_car):                                         
        # (Eq. 3.12 B&T) The canopy resistance is determined by three resistances in series.
        return r_bnd + r_stm + r_car

    def r_car(self, c_car_1, c_car_2, c_car_3, T_crop):                           
        # (Eq. 3.13 B&T) The carboxylation resistance.
        return 1 / (c_car_1 * T_crop ** 2 + c_car_2 * T_crop + c_car_3)

    def r_bnd(self, l, u_inf, LAI):
        # (Eq. 3.14 B&T) The boundary layer resistance.
        return 350 * (l / u_inf)**0.5 * LAI**(-1)                   # l = leaf_diameter

    def r_stm(self, PPFD):
        # (Eq. 3.15 B&T) The stomatal resistance.
        return 60 * (1500+PPFD) / (200+PPFD)
    
    def X_dw(self, X_fw, c_d, c_tau): 
        # (Eq. 3.16 B&T) The dry weight. This is equation 3.16 turned around, because we want the output to be the dry weight of a plant, with fresh weight of the initial shoot as input. 
        return X_fw * c_d / (1 - c_tau)
        #Returns the total crop or a single plant depending on the input


    def update_values(self, X_sdw: float, X_nsdw: float):
        self.X_sdw = X_sdw
        self.X_nsdw = X_nsdw            
        self.X_dw = self.X_dw(self.X_nsdw, self.X_sdw)                       #X_sdw + X_nsdw (All plants)
        self.dry_weight_per_plant = self.X_dw / self.planting_crop_density   #Single plant. Used in env.py
        self.LAI = self.LAI(self.c_lar, self.c_tau, self.X_sdw)
        self.CAC = self.CAC(self.c_K, self.LAI)

    def biomass_ode(self, X_nsdw, X_sdw, PPFD):
        #Implementing all the equations
        r_stm = self.r_stm(PPFD)
        r_bnd = self.r_bnd(self.leaf_diameter, self.air_vel, self.LAI)
        r_car = self.r_car(self.c_car_1, self.c_car_2, self.c_car_3, self.T_crop)
        r_CO2 = self.r_CO2(r_bnd, r_stm, r_car)
        gamma = self.gamma(self.c_gamma, self.c_Q10_Gamma, self.T_crop)
        f_sat = self.f_sat(self.c_w, self.CO2_in, gamma, r_CO2)
        epsilon = self.epsilon(self.c_epsilon, self.CO2_in, gamma)
        U_par = self.U_par(self.c_p, PPFD)
        f_phot_max = self.f_phot_max(epsilon, U_par, f_sat)
        f_phot = self.f_phot_ODE(f_phot_max, self.CAC)
        r_gr = self.r_gr(self.c_gr_max, self.X_nsdw, self.X_sdw, self.c_Q10_gr, self.T_crop)
        f_resp = self.f_resp(self.c_resp_sht, self.c_tau, self.X_sdw, self.c_resp_rt, self.c_Q10_resp, self.T_crop)
        dX_sdw = self.rate_of_X_sdw(r_gr, self.X_sdw)
        dX_nsdw = self.rate_of_X_nsdw(self.c_alpha, f_phot, r_gr, self.X_sdw, f_resp, self.c_beta)
        return dX_nsdw, dX_sdw

    def combined_ODE(self, state, control_inputs, data):
        T_in, Chi_in, CO2_in, T_env, T_sup, Chi_sup, X_nsdw, X_sdw = state
        PPFD = control_inputs[6]
        
        dNS_dt, dS_dt = self.biomass_ode(X_nsdw, X_sdw, PPFD)
        return np.array([dNS_dt, dS_dt])

    def print_attributes(self, *args):
        if args:
            for attr_name in args:
                print(f"{attr_name}: {getattr(self, attr_name, 'Attribute not found')}")
        else:
            for attr, value in vars(self).items():
                print(f"{attr}: {value}")