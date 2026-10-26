from geometry import AxisymmetricSection, FinSet
from kraken import Kraken
from mass_model import FinMassComponent, FixedMassComponent, MotorMassComponent, ShellMassComponent
from vehicle import Vehicle, VehicleStage


def build_example_vehicle():
    def make_shell(length, radius_at_axial_distance):
        return ShellMassComponent(
            section=AxisymmetricSection(
                length=length, radius_at_axial_distance=radius_at_axial_distance,
            ),
            wall_thickness=0.003,
            material_density=1600.0,
            # Vehicle computes the actual shell positions from their order.
            forward_axial_position=0.0,
        )

    nose_cone = make_shell(0.3, lambda distance: 0.06 * distance / 0.3)
    upper_stage_body = make_shell(1.0, lambda distance: 0.06)
    transition = make_shell(0.2, lambda distance: 0.06 + 0.025 * distance / 0.2)
    booster_body = make_shell(1.0, lambda distance: 0.085)

    upper_stage_fins = FinMassComponent(
        fin_set=FinSet(
            number_of_fins=4, fin_root_chord=0.2, fin_tip_chord=0.1,
            fin_span=0.1, fin_leading_edge_sweep_distance=0.05,
            fin_leading_edge_axial_position=1.0, body_diameter_at_fin_attachment=0.12,
        ),
        fin_thickness=0.003, material_density=1600.0,
    )
    booster_fins = FinMassComponent(
        fin_set=FinSet(
            number_of_fins=3, fin_root_chord=0.3, fin_tip_chord=0.1,
            fin_span=0.15, fin_leading_edge_sweep_distance=0.1,
            fin_leading_edge_axial_position=2.0, body_diameter_at_fin_attachment=0.17,
        ),
        fin_thickness=0.003, material_density=1600.0,
    )
    upper_stage = VehicleStage(
        body_sections=(upper_stage_body,),
        fin_sets=(upper_stage_fins,),
        fixed_components=(FixedMassComponent(mass=1.0, center_of_mass=0.4),),
        motor=MotorMassComponent(
            motor=Kraken(propellant_mass=2.0),
            forward_axial_position=0.6, ignition_time=12.0,
        ),
    )
    booster_stage = VehicleStage(
        body_sections=(transition, booster_body),
        fin_sets=(booster_fins,),
        motor=MotorMassComponent(
            motor=Kraken(propellant_mass=3.0),
            forward_axial_position=1.6, ignition_time=0.0,
        ),
    )
    return Vehicle(
        nose_cone=nose_cone,
        upper_stage=upper_stage,
        booster_stage=booster_stage,
        separation_time=11.0,
        stacked_reference_diameter=0.17,
        upper_stage_reference_diameter=0.12,
    )


def main():
    vehicle = build_example_vehicle()

    # Include the instant before separation, the exact event, and both burns.
    # These would be simulation supplied in the future.
    flight_times = [0.0, 4.89, 9.78, 10.999999, 11.0, 12.0, 16.89, 21.78]
    mach_numbers = [0.0, 0.3, 0.6, 0.6, 0.6, 0.6, 0.8, 0.7]

    mass = vehicle.total_mass(flight_times)
    center_of_mass = vehicle.center_of_mass(flight_times)
    center_of_pressure = vehicle.center_of_pressure(flight_times, mach_numbers)
    stability_margin = vehicle.stability_margin(flight_times, mach_numbers)

    print("Time [s] | Mach | Mass [kg] | CG [m] | CP [m] | Margin [calibers]")
    for values in zip(flight_times, mach_numbers, mass, center_of_mass,
                      center_of_pressure, stability_margin):
        print("{:8.6f} | {:.2f} | {:9.4f} | {:.4f} | {:.4f} | {:.4f}".format(*values))

if __name__ == "__main__":
    main()
