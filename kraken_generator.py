# NOTES:
# 1. All units are in SI unless otherwise specified.
# 2. Naming scheme is as follows: <motor_type>-<total_impulse>-<thrust>. As an example, a high altitude 
#    optimized motor with an impulse of 10100 kNs and average thrust of 800N would be named "KVac-10-800". 
#    A similar thrust sea level optimized motor with an impulse of 50600 Ns would be named "Kraken-51-800".
# 3. Sections are divided into (INPUT) and (OUTPUT) parameters. The (INPUT) parameters are the 
#    ones that the user can change, while the (OUTPUT) parameters are calculated based on the
#    (INPUT) parameters.

# (CONSTANTS) General Constants
R_universal = 8.314
g0 = 9.80665
propellant_OD = 3.75 / 39.4
propellant_ID = 0.875 / 39.4
cross_sectional_area = 3.14159 * ((propellant_OD / 2) ** 2 - (propellant_ID / 2) ** 2)
asl_pressure = 1.0135 # Note that this is in bar, not Pa.

nozzle_mass = 0.250 # KRAKEN Fusion CAD Model
forward_closure_mass = 0.287 # KRAKEN Fusion CAD Model
liner_mass_per_unit_length = 0.677 # KRAKEN Fusion CAD Model
casing_mass_per_unit_length = 1.365 # KRAKEN Fusion CAD Model
nozzle_closure_length = 0.0761 # KRAKEN Fusion CAD Model

dt = 0.02
motor_diameter = 4.0 / 39.4 # KRAKEN Sizing Sheet w/ SILVERSAND Properties.

# (INPUT) General Overview
sea_level_optimized = False
simulator = "RASAERO" # Or "RASAERO".

# (INPUT) Motor Parameters
propellant_density = 1611 # SILVERSAND-02
total_impulse = 14000
pressure_exit = 0.1 # Note that this is in bar, not Pa.
isp = 255.8 # KRAKEN Sizing Sheet w/ SILVERSAND Properties, Pe=1 bar.
burn_duration = 8.3 # KRAKEN Sizing Sheet w/ SILVERSAND Properties.
nozzle_exit_diameter = 3.908 / 39.4 # KRAKEN Sizing Sheet w/ SILVERSAND Properties.

# (OUTPUT) Motor Properties
propellant_mass = total_impulse / (isp * g0)
propellant_volume = propellant_mass / propellant_density
exit_area = 3.14159 * (nozzle_exit_diameter / 2) ** 2

grain_length = propellant_volume / cross_sectional_area
total_motor_length = grain_length + nozzle_closure_length

casing_mass = casing_mass_per_unit_length * total_motor_length
liner_mass = liner_mass_per_unit_length * grain_length
dry_mass = casing_mass + liner_mass + nozzle_mass + forward_closure_mass
wet_mass = dry_mass + propellant_mass
average_thrust = total_impulse / burn_duration
average_thrust_asl = average_thrust + 100000 * (pressure_exit - asl_pressure) * exit_area

# (OUTPUT) .ENG File Generation
motor_type = "KVac"
if sea_level_optimized:
        motor_type = "Kraken"
impulse_KNs = round(total_impulse / 1000)
thrust_N_string = str(round(average_thrust))

motor_name = f"{motor_type}-{impulse_KNs}-{thrust_N_string}"
diameter_mm = str(round(motor_diameter * 1000, 2))
length_mm = str(round(total_motor_length * 1000, 2))
delay = "P"
propellant_mass_string = str(round(propellant_mass, 3))
total_mass_string = str(round(wet_mass, 3))
manufacturer = "SLRT"
exit_diameter_mm = str(round(nozzle_exit_diameter * 1000, 1))

times = (dt, burn_duration, burn_duration + dt)
thrust = (average_thrust, average_thrust, 0)
if simulator == "RASAERO":
    thrust = (average_thrust_asl, average_thrust_asl, 0)

with open(f"MOTORS/RASAERO/{motor_name}.eng", "x") as f:
    f.write(f"; {manufacturer} {motor_name}\n")
    f.write(f"; Exit diameter of {exit_diameter_mm} mm\n")
    f.write(f"{motor_name} {diameter_mm} {length_mm} {delay} {propellant_mass_string} {total_mass_string} {manufacturer}\n")
    for i in range(len(times)):
        f.write(f"     {round(times[i], 3)} {round(thrust[i], 1)}\n")
    f.write(";\n")
    f.close()

print(f".ENG file for {motor_name} successfully generated.")


    