"""Export per-component mass properties from a Fusion 360 design to JSON.

HOW TO RUN
----------
1. Open the rocket design in Fusion 360.
2. Utilities tab -> ADD-INS -> Scripts and Add-Ins  (or Shift+S)
3. Scripts tab -> green "+" next to "My Scripts" -> select THIS FOLDER
4. Select "fusion_export_mass_properties" -> Run
5. It writes mass_properties_<document name>.json next to this script.

RUN IT THREE TIMES — once with the full assembly open, once with the stage 1
design open, once with stage 2. Output is named after the active document so the
three do not overwrite each other. All three are wanted:
  * full assembly -> stack CG / inertia for the mated static-margin check
  * stage 1 alone -> booster mass properties
  * stage 2 alone -> sustainer properties, the post-separation configuration
                     whose static margin is usually the binding constraint
Stage 1 + stage 2 should reconcile against the full assembly (allowing for
interstage hardware). That reconciliation is a free error check on the data: if
they do not add up, a material is unassigned or a part is double-counted.

WHAT IT EXPORTS
---------------
For every component occurrence: mass, center of mass, inertia tensor, bounding
box, and the assigned material (including Young's / shear modulus when the
material library provides them).

IMPORTANT CAVEATS — read before trusting the numbers
----------------------------------------------------
* Materials MUST be correctly assigned in Fusion. Default "Steel" on a
  fiberglass airframe produces garbage. Sanity-check `total_mass_kg` against
  your expected vehicle mass before using this.
* The Kraken motor is modelled separately by kraken.py. Either delete/suppress
  motor components before running, or name them with "motor" / "kraken" /
  "propellant" so the importer can exclude them and avoid double-counting.
* CAD systematically UNDER-estimates real mass: fasteners, epoxy fillets,
  wiring, potting, tape, chute packing. Expect 5-15% low. Weighed masses of
  parts that physically exist beat CAD every time.
* Fusion API internal units are cm and kg. This script converts lengths to m
  and moments of inertia to kg*m^2.
* Moments of inertia from getXYZMomentsOfInertia are reported about the world
  origin. `moi_datum` records this.

COORDINATE FRAMES -- IMPORTANT
------------------------------
Occurrence `cg_m` is in WORLD coordinates and can be used directly.
Body `cg_m` is in the BODY's LOCAL frame, so body CGs must NOT be summed into a
vehicle CG -- verified by comparing a single-body occurrence's CG (which matches
its world bounding box) against its body's CG (which sits near its own origin).
Body masses ARE frame-independent and safe to sum.

To get world-frame body CGs, a future version would need to transform each body's
centre of mass by its occurrence's `transform2` matrix. Until then, use
occurrence-level CG for anything positional, and body-level data only for mass
and material.
"""

import json
import os
import traceback

import adsk.core
import adsk.fusion

CM_TO_M = 0.01
KGCM2_TO_KGM2 = 1.0e-4

# Material properties worth pulling for the flutter module. Fusion's property
# names vary by material library, so several spellings are tried.
MATERIAL_PROPERTY_ALIASES = {
    "youngs_modulus": ["Young's Modulus", "Youngs Modulus", "Young Modulus"],
    "poissons_ratio": ["Poisson's Ratio", "Poissons Ratio"],
    "shear_modulus": ["Shear Modulus"],
    "density": ["Density"],
    "yield_strength": ["Yield Strength"],
    "tensile_strength": ["Tensile Strength", "Ultimate Tensile Strength"],
}


def _safe(fn, default=None):
    """Fusion's API raises liberally on incomplete designs; degrade instead."""
    try:
        return fn()
    except Exception:
        return default


def _material_info(material):
    if material is None:
        return None
    info = {"name": _safe(lambda: material.name)}
    props = _safe(lambda: material.materialProperties)
    if props is None:
        return info
    for key, aliases in MATERIAL_PROPERTY_ALIASES.items():
        for alias in aliases:
            prop = _safe(lambda: props.itemByName(alias))
            if prop is None:
                continue
            value = _safe(lambda: prop.value)
            if value is not None:
                info[key] = value
                info[key + "_units"] = _safe(lambda: prop.units, "")
                break
    return info


def _physical_properties(entity):
    """Return mass / CG / inertia for a component, occurrence, or body."""
    pp = _safe(lambda: entity.physicalProperties)
    if pp is None:
        return None

    out = {
        "mass_kg": _safe(lambda: pp.mass),
        "volume_m3": _safe(lambda: pp.volume * CM_TO_M**3),
        "area_m2": _safe(lambda: pp.area * CM_TO_M**2),
        # Fusion PhysicalProperties.density is in kg/cm^3 (internal units are cm
        # and kg), so the factor to kg/m^3 is 1e6, NOT 1e3. Verified against
        # known materials: Al 6061 -> 2700, GFRP -> 1750, Steel -> 7850.
        "density_kg_m3": _safe(lambda: pp.density * 1.0e6),
    }

    cg = _safe(lambda: pp.centerOfMass)
    if cg is not None:
        out["cg_m"] = {
            "x": cg.x * CM_TO_M,
            "y": cg.y * CM_TO_M,
            "z": cg.z * CM_TO_M,
        }

    moi = _safe(lambda: pp.getXYZMomentsOfInertia())
    if moi is not None and moi[0]:
        _, xx, yy, zz, xy, yz, xz = moi
        out["moi_kg_m2"] = {
            "xx": xx * KGCM2_TO_KGM2,
            "yy": yy * KGCM2_TO_KGM2,
            "zz": zz * KGCM2_TO_KGM2,
            "xy": xy * KGCM2_TO_KGM2,
            "yz": yz * KGCM2_TO_KGM2,
            "xz": xz * KGCM2_TO_KGM2,
        }
        out["moi_datum"] = "world_origin"

    return out


def _bounding_box(entity):
    bb = _safe(lambda: entity.boundingBox)
    if bb is None:
        return None
    return {
        "min_m": {
            "x": bb.minPoint.x * CM_TO_M,
            "y": bb.minPoint.y * CM_TO_M,
            "z": bb.minPoint.z * CM_TO_M,
        },
        "max_m": {
            "x": bb.maxPoint.x * CM_TO_M,
            "y": bb.maxPoint.y * CM_TO_M,
            "z": bb.maxPoint.z * CM_TO_M,
        },
        "extent_m": {
            "x": (bb.maxPoint.x - bb.minPoint.x) * CM_TO_M,
            "y": (bb.maxPoint.y - bb.minPoint.y) * CM_TO_M,
            "z": (bb.maxPoint.z - bb.minPoint.z) * CM_TO_M,
        },
    }


def _bodies(component):
    records = []
    bodies = _safe(lambda: component.bRepBodies)
    if bodies is None:
        return records
    for body in bodies:
        if not _safe(lambda: body.isSolid, False):
            continue
        records.append(
            {
                "name": _safe(lambda: body.name),
                "is_visible": _safe(lambda: body.isVisible),
                "material": _material_info(_safe(lambda: body.material)),
                "properties": _physical_properties(body),
                "bounding_box": _bounding_box(body),
            }
        )
    return records


def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface
        design = adsk.fusion.Design.cast(app.activeProduct)
        if design is None:
            ui.messageBox("No active Fusion design. Open the rocket and retry.")
            return

        root = design.rootComponent

        export = {
            "schema": "fusion_mass_properties/1",
            "document_name": _safe(lambda: app.activeDocument.name),
            "units": {
                "length": "m",
                "mass": "kg",
                "moment_of_inertia": "kg*m^2",
                "note": "Converted from Fusion internal cm / kg.",
            },
            "root": {
                "name": _safe(lambda: root.name),
                "properties": _physical_properties(root),
                "bounding_box": _bounding_box(root),
            },
            "occurrences": [],
            "root_bodies": _bodies(root),
        }

        # The vehicle long axis is whichever bounding-box extent dominates.
        # Recorded so the importer does not have to guess.
        bb = export["root"].get("bounding_box")
        if bb:
            extent = bb["extent_m"]
            export["inferred_long_axis"] = max(extent, key=extent.get)
            export["inferred_overall_length_m"] = max(extent.values())

        occurrences = _safe(lambda: root.allOccurrences, [])
        for occ in occurrences:
            component = _safe(lambda: occ.component)
            export["occurrences"].append(
                {
                    "name": _safe(lambda: occ.name),
                    "component_name": _safe(lambda: component.name) if component else None,
                    "full_path": _safe(lambda: occ.fullPathName),
                    "is_visible": _safe(lambda: occ.isVisible),
                    "is_suppressed": _safe(lambda: occ.isSuppressed),
                    "is_reference": _safe(lambda: occ.isReferencedComponent),
                    "properties": _physical_properties(occ),
                    "bounding_box": _bounding_box(occ),
                    "bodies": _bodies(component) if component else [],
                }
            )

        masses = [
            o["properties"]["mass_kg"]
            for o in export["occurrences"]
            if o["properties"] and o["properties"].get("mass_kg")
        ]
        export["component_count"] = len(export["occurrences"])
        export["summed_occurrence_mass_kg"] = sum(masses)
        root_props = export["root"].get("properties") or {}
        export["total_mass_kg"] = root_props.get("mass_kg")
        export["mass_check_note"] = (
            "summed_occurrence_mass_kg double-counts nested assemblies. "
            "Trust total_mass_kg; use the per-occurrence values for the buildup."
        )

        # Name the file after the active document so the full assembly, stage 1,
        # and stage 2 exports do not overwrite each other.
        doc_name = export["document_name"] or "unknown_document"
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in doc_name)
        out_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "mass_properties_{}.json".format(safe_name),
        )
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(export, handle, indent=2)

        total = export["total_mass_kg"]
        ui.messageBox(
            "Exported {} occurrences.\n\n"
            "Total mass: {}\n"
            "Long axis: {}\n"
            "Overall length: {} m\n\n"
            "Written to:\n{}\n\n"
            "CHECK the total mass against your expected vehicle mass before "
            "using this. A wrong number here means materials are not assigned "
            "correctly in Fusion.".format(
                export["component_count"],
                "{:.3f} kg".format(total) if total else "unavailable",
                export.get("inferred_long_axis", "unknown"),
                "{:.4f}".format(export.get("inferred_overall_length_m", 0.0)),
                out_path,
            )
        )

    except Exception:
        if ui:
            ui.messageBox("Script failed:\n{}".format(traceback.format_exc()))
        else:
            raise
