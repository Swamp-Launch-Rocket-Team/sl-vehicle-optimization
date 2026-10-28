# NOTES:
# 1. All units are in SI unless otherwise specified.


import math

import numpy as np

class Kraken:
    # PHYSICAL CONSTANTS
    g0 = 9.80665
    R_universal = 8.314

    # PROPELLANT PROPERTIES
    p_density = 1612
    p_flame_temperature = 2528
    p_molecular_weight = 22.06 # g/mol
    p_gamma = 1.22
    p_specific_gas_constant = 1000 * 8.314 / p_molecular_weight

    # MOTOR PROPERIES
    motor_diameter = 4.0 / 39.4
    propellant_OD = 3.75 / 39.4
    propellant_ID = 1.0 / 39.4
    cross_sectional_area = np.pi * ((propellant_OD / 2) ** 2 - (propellant_ID / 2) ** 2)

    nozzle_closure_length = 0.0761
    forward_closure_length = 0.0761
    nozzle_mass = 0.171
    forward_closure_mass = 0.174
    liner_mass_per_unit_length = 0.677
    casing_mass_per_unit_length = 1.365

    chamber_pressure = 38 # bar
    burn_time = 9.78 # s

    # EFFICIENCIES
    c_star_efficency = 0.95
    c_thrust_efficency = 0.95

    def __init__(self, propellant_mass):
        self.p_mass = propellant_mass
        self.p_volume = self.p_mass / self.p_density
        self.p_length = self.p_volume / self.cross_sectional_area
        self.motor_length = self.p_length + self.nozzle_closure_length + self.forward_closure_length
        self.casing_mass = self.casing_mass_per_unit_length * self.motor_length
        self.liner_mass = self.liner_mass_per_unit_length * self.p_length
        self.dry_mass = self.casing_mass + self.liner_mass + self.nozzle_mass + self.forward_closure_mass
        self.wet_mass = self.dry_mass + self.p_mass
        self.center_of_mass = self.motor_length / 2.0
        self.reset_thrust_properties()

    def reset_thrust_properties(self):
        self.momentum_thrust = None
        self.exit_pressure = None  # bar
        self.exit_area = None  # m^2

    def initialize_thrust(self, pressure_pa):
        (self.momentum_thrust, self.burn_time, self.exit_pressure,
         self.exit_area) = self.get_thrust_properties(pressure_pa / 100000.0)

    def get_mass_properties(self):
        return (self.dry_mass, self.wet_mass, self.center_of_mass, self.motor_length)

    # pressure_a - ambient pressure [bar]
    # pressure_e - nozzle exit pressure [bar]
    def get_thrust_properties(self, pressure_a):
        pressure_e = pressure_a
        c_star_ideal = math.sqrt(self.p_gamma*self.p_specific_gas_constant*self.p_flame_temperature)/(self.p_gamma*math.sqrt(math.pow(2/(self.p_gamma+1), (self.p_gamma+1)/(self.p_gamma-1))))
        c_f_ideal = (2*math.pow(self.p_gamma, 2)/(self.p_gamma-1)*math.pow(2/(self.p_gamma+1),(self.p_gamma+1)/(self.p_gamma-1))*(1-math.pow(pressure_e/self.chamber_pressure, (self.p_gamma-1)/self.p_gamma)))**0.5
        expansion_ratio = math.pow(2/(self.p_gamma+1), 1/(self.p_gamma-1)) * math.pow(self.chamber_pressure / pressure_e, 1/self.p_gamma) / math.sqrt((self.p_gamma+1)/(self.p_gamma-1) * (1 - math.pow(pressure_e/self.chamber_pressure, (self.p_gamma-1)/self.p_gamma)))

        mass_flow_rate = self.p_mass / self.burn_time
        c_star = self.c_star_efficency * c_star_ideal
        c_f = self.c_thrust_efficency * c_f_ideal

        throat_area = mass_flow_rate * c_star / (self.chamber_pressure * 100000)
        exit_area = expansion_ratio * throat_area
        exit_velocity = c_f * c_star

        momentum_thrust = mass_flow_rate * exit_velocity

        return (momentum_thrust, self.burn_time, pressure_e, exit_area)




    def thrust(self, elapsed_time, pressure_pa):
        """Burn thrust [N] with fixed nozzle properties and ambient pressure [Pa]."""
        if 0 <= elapsed_time < self.burn_time:
            if self.momentum_thrust is None:
                raise RuntimeError("Initialize thrust properties at ignition before evaluating a burn.")
            pressure_thrust = (self.exit_pressure * 100000.0 - pressure_pa) * self.exit_area
            return self.momentum_thrust + pressure_thrust
        return 0.0

    def remaining_propellant_mass(self, elapsed_time):
        return self.p_mass * (1.0 - np.clip(np.asarray(elapsed_time) / self.burn_time, 0, 1))
