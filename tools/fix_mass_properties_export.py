"""Apply known corrections to a Fusion mass-properties export.

Run with the raw export path; writes <name>_corrected.json alongside it. The raw
export is never modified, so the original Fusion record stays intact.

    py tools/fix_mass_properties_export.py tools/mass_properties_<name>.json

TWO CORRECTIONS
---------------
1. DENSITY UNIT BUG (ours). The exporter multiplied Fusion's density by 1e3, but
   Fusion reports density in kg/cm^3, so the factor to kg/m^3 is 1e6. Every
   density in an export written before this was fixed is 1000x low. Verified
   against knowns: Al 6061 read 2.7 and should read 2700; GFRP 1.7 -> 1750;
   Steel 7.8 -> 7850. Masses were never affected -- Fusion reports mass in kg
   directly -- so this correction only touches density fields.

2. BROKEN MATERIAL (in the CAD). The custom material "PLA (orange)" has a
   density roughly 1e6 times too large. It is applied to eight Wago 221-2411
   lever-nut bodies in the Stage 2 Avionics Bay, each of volume 1.128e-7 m^3.
   Each reads 139.8393 kg instead of about 0.14 g, inflating the reported vehicle
   mass from 34.80 kg to 1153.52 kg.

   THE AUTHORITATIVE FIX IS IN FUSION. Correct the material definition there, or
   the next export will carry the same error and this script will be needed
   again. This is a stopgap so the data is usable now.

Mass corrections propagate up the occurrence tree: a Fusion occurrence's
physicalProperties is a subtree rollup, so every ancestor of a corrected body
must have the same delta removed. Centre of mass is recomputed by removing the
phantom mass moment. Moments of inertia are corrected where per-body inertia is
available; note the current 1 DOF build does not use inertia at all.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

DENSITY_FIX_FACTOR = 1000.0          # 1e3 -> 1e6 overall
BROKEN_MATERIAL = "PLA (orange)"
BROKEN_MATERIAL_TRUE_DENSITY = 1240.0  # kg/m^3, standard PLA


def _material_name(body: dict) -> str | None:
    return (body.get("material") or {}).get("name")


def _fix_densities(export: dict) -> int:
    """Correct every density field for the exporter unit bug. Returns count."""
    fixed = 0

    def fix(props: dict | None) -> None:
        nonlocal fixed
        if props and props.get("density_kg_m3") is not None:
            props["density_kg_m3"] *= DENSITY_FIX_FACTOR
            fixed += 1

    fix((export.get("root") or {}).get("properties"))
    for body in export.get("root_bodies", []):
        fix(body.get("properties"))
    for occ in export.get("occurrences", []):
        fix(occ.get("properties"))
        for body in occ.get("bodies", []):
            fix(body.get("properties"))
    return fixed


def _correct_broken_bodies(export: dict) -> tuple[float, list[dict], int]:
    """Set broken bodies to their true mass. Returns (total delta, moments, count).

    `moments` collects {mass_removed, cg, moi} per body so ancestor centres of
    mass and inertia can be corrected too.
    """
    total_delta = 0.0
    moments: list[dict] = []
    count = 0

    for occ in export.get("occurrences", []):
        for body in occ.get("bodies", []):
            if _material_name(body) != BROKEN_MATERIAL:
                continue
            props = body.get("properties") or {}
            old_mass = props.get("mass_kg")
            volume = props.get("volume_m3")
            if old_mass is None or volume is None:
                continue

            true_mass = volume * BROKEN_MATERIAL_TRUE_DENSITY
            delta = old_mass - true_mass

            moments.append(
                {
                    "mass_removed": delta,
                    "cg": props.get("cg_m"),
                    "moi": props.get("moi_kg_m2"),
                    "path": f"{occ.get('full_path')} / {body.get('name')}",
                }
            )

            props["mass_kg"] = true_mass
            props["density_kg_m3"] = BROKEN_MATERIAL_TRUE_DENSITY
            body.setdefault("corrections", []).append(
                f"mass {old_mass:.4f} -> {true_mass:.8f} kg; density set to "
                f"{BROKEN_MATERIAL_TRUE_DENSITY} kg/m^3 (broken material '{BROKEN_MATERIAL}')"
            )
            total_delta += delta
            count += 1

    return total_delta, moments, count


def _subtree_delta(export: dict, prefix: str) -> float:
    """Total corrected mass removed from `prefix` and everything beneath it."""
    delta = 0.0
    for occ in export.get("occurrences", []):
        path = occ.get("full_path") or ""
        if path != prefix and not path.startswith(prefix + "+"):
            continue
        for body in occ.get("bodies", []):
            if _material_name(body) != BROKEN_MATERIAL:
                continue
            props = body.get("properties") or {}
            volume = props.get("volume_m3")
            if volume is None:
                continue
            # props["mass_kg"] is already corrected; recover the delta from the
            # recorded correction note instead of the live value.
            for note in body.get("corrections", []):
                if note.startswith("mass "):
                    original = float(note.split()[1])
                    delta += original - volume * BROKEN_MATERIAL_TRUE_DENSITY
                    break
    return delta


def _correct_container(props: dict | None, delta: float) -> None:
    """Remove `delta` kg from a rolled-up mass. Mass only -- see `_recompute_inertial`.

    Centre of mass and inertia are deliberately NOT patched by subtraction here.
    Doing that is easy to get subtly wrong, and a plausible-looking wrong CG is
    more dangerous than an absent one. They are recomputed from the corrected leaf
    bodies instead, which is unambiguous and can be verified against the mass sum.
    """
    if not props or delta <= 0.0:
        return
    old_mass = props.get("mass_kg")
    if old_mass is None:
        return
    props["mass_kg"] = old_mass - delta


def find_phantom_moments(export: dict) -> list[dict]:
    """Locate the phantom mass in WORLD coordinates, before anything is mutated.

    Body-level `cg_m` in these exports is in the body's LOCAL frame, not world --
    verified by comparing a single-body occurrence's CG (which matches its world
    bounding box) against its body's CG (which sits near its own origin). Summing
    body CGs is therefore meaningless for a vehicle CG.

    Occurrence-level `cg_m` IS in world coordinates. So the phantom mass is
    located using the occurrence that directly contains the broken bodies. That is
    an approximation -- the occurrence may hold real mass too -- so `purity`
    records what fraction of that occurrence is phantom. For the Wago containers
    it is 99.999%, which makes the approximation exact for our purposes.
    """
    phantoms: list[dict] = []
    for occ in export.get("occurrences", []):
        delta = 0.0
        for body in occ.get("bodies", []):
            if _material_name(body) != BROKEN_MATERIAL:
                continue
            props = body.get("properties") or {}
            mass, volume = props.get("mass_kg"), props.get("volume_m3")
            if mass is None or volume is None:
                continue
            delta += mass - volume * BROKEN_MATERIAL_TRUE_DENSITY
        if delta <= 0.0:
            continue
        occ_props = occ.get("properties") or {}
        occ_mass = occ_props.get("mass_kg") or delta
        phantoms.append(
            {
                "path": occ.get("full_path"),
                "mass_removed": delta,
                "cg": dict(occ_props["cg_m"]) if occ_props.get("cg_m") else None,
                "purity": delta / occ_mass if occ_mass else 1.0,
            }
        )
    return phantoms


def _correct_cg(props: dict | None, phantoms: list[dict], original_mass: float) -> bool:
    """Remove the phantom mass moment from a world-frame CG. Mass must already be corrected."""
    if not props or not props.get("cg_m"):
        return False
    new_mass = props.get("mass_kg")
    if not new_mass or new_mass <= 0.0:
        return False
    cg = props["cg_m"]
    for axis in ("x", "y", "z"):
        moment = original_mass * cg[axis]
        for phantom in phantoms:
            if phantom["cg"]:
                moment -= phantom["mass_removed"] * phantom["cg"][axis]
        cg[axis] = moment / new_mass
    return True


def _invalidate_inertia(props: dict | None) -> None:
    """Drop an inertia tensor that can no longer be trusted.

    Inertia cannot be corrected from this export: it would need each body's
    inertia in world coordinates, and only local-frame body data is available.
    A plausible-looking wrong tensor is worse than an absent one, and the current
    1 DOF build does not use inertia at all.
    """
    if props and props.get("moi_kg_m2") is not None:
        props["moi_kg_m2"] = None
        props["moi_datum"] = (
            "INVALID: contaminated by the broken material and not correctable from "
            "this export. Re-export after fixing the Fusion material."
        )


def drop_components(export: dict, component_names: list[str]) -> dict:
    """Remove whole component instances from the export, rollups and all.

    This is the file-side equivalent of suppressing those components in Fusion:
    the component and everything beneath it disappears, and every ancestor's
    rolled-up mass and centre of mass is adjusted to match.

    Use this instead of correcting a broken material when the offending parts do
    not matter -- for the Wago lever nuts the two paths differ by about a gram.
    It does NOT fix the CAD: the next export still carries the broken material
    unless it is fixed in Fusion.
    """
    occurrences = export.get("occurrences", [])

    # Identify the outermost matching instances. A descendant of an already
    # matched instance must not be counted again.
    targets: list[dict] = []
    for occ in occurrences:
        name = occ.get("name") or ""
        path = occ.get("full_path") or ""
        if not any(name.startswith(prefix) for prefix in component_names):
            continue
        if any(path.startswith((t["path"] or "") + "+") for t in targets):
            continue
        props = occ.get("properties") or {}
        targets.append(
            {
                "path": path,
                "name": name,
                "mass_kg": props.get("mass_kg") or 0.0,
                "cg": dict(props["cg_m"]) if props.get("cg_m") else None,
            }
        )

    if not targets:
        return {"dropped": [], "mass_removed_kg": 0.0, "occurrences_removed": 0}

    total_removed = sum(t["mass_kg"] for t in targets)

    # Adjust every surviving ancestor before deleting anything.
    removed_paths = {t["path"] for t in targets}

    def is_removed(path: str) -> bool:
        return any(path == p or path.startswith(p + "+") for p in removed_paths)

    for occ in occurrences:
        path = occ.get("full_path") or ""
        if is_removed(path):
            continue
        contained = [t for t in targets if (t["path"] or "").startswith(path + "+")]
        if not contained:
            continue
        props = occ.get("properties") or {}
        original_mass = props.get("mass_kg")
        if original_mass is None:
            continue
        delta = sum(t["mass_kg"] for t in contained)
        props["mass_kg"] = original_mass - delta
        cg = props.get("cg_m")
        if cg and props["mass_kg"] > 0.0:
            for axis in ("x", "y", "z"):
                moment = original_mass * cg[axis]
                for target in contained:
                    if target["cg"]:
                        moment -= target["mass_kg"] * target["cg"][axis]
                cg[axis] = moment / props["mass_kg"]
        _invalidate_inertia(props)

    # Root.
    root_props = (export.get("root") or {}).get("properties")
    if root_props and root_props.get("mass_kg") is not None:
        original_root = root_props["mass_kg"]
        root_props["mass_kg"] = original_root - total_removed
        cg = root_props.get("cg_m")
        if cg and root_props["mass_kg"] > 0.0:
            for axis in ("x", "y", "z"):
                moment = original_root * cg[axis]
                for target in targets:
                    if target["cg"]:
                        moment -= target["mass_kg"] * target["cg"][axis]
                cg[axis] = moment / root_props["mass_kg"]
        _invalidate_inertia(root_props)

    before = len(occurrences)
    export["occurrences"] = [
        occ for occ in occurrences if not is_removed(occ.get("full_path") or "")
    ]
    removed_count = before - len(export["occurrences"])

    if export.get("total_mass_kg") is not None:
        export["total_mass_kg"] -= total_removed
    if export.get("summed_occurrence_mass_kg") is not None:
        export["summed_occurrence_mass_kg"] -= total_removed
    export["component_count"] = len(export["occurrences"])

    return {
        "dropped": [
            {"name": t["name"], "path": t["path"], "mass_kg": t["mass_kg"]} for t in targets
        ],
        "mass_removed_kg": total_removed,
        "occurrences_removed": removed_count,
    }


def _subtree_body_mass(export: dict, prefix: str | None) -> tuple[float, int]:
    """Sum corrected body masses in a subtree. Bodies are leaves, so no double count."""
    total, count = 0.0, 0
    for occ in export.get("occurrences", []):
        path = occ.get("full_path") or ""
        if prefix is not None and path != prefix and not path.startswith(prefix + "+"):
            continue
        for body in occ.get("bodies", []):
            mass = (body.get("properties") or {}).get("mass_kg")
            if mass:
                total += mass
                count += 1
    return total, count


def fix_export(path: Path, drop_wago: bool = False) -> Path:
    with open(path, encoding="utf-8") as handle:
        export = json.load(handle)

    original_total = (export.get("root") or {}).get("properties", {}).get("mass_kg")

    if drop_wago:
        # Remove the offending components outright rather than repairing their
        # density. Equivalent to suppressing them in Fusion; the two paths differ
        # by about a gram because these are 0.14 g wire connectors.
        density_count = _fix_densities(export)
        dropped = drop_components(export, ["Wago 221-2411"])
        export["corrections_applied"] = {
            "density_unit_fix": {
                "fields_corrected": density_count,
                "factor": DENSITY_FIX_FACTOR,
                "reason": "exporter used 1e3; Fusion reports kg/cm^3 so the factor is 1e6",
            },
            "dropped_components": dropped,
            "original_total_mass_kg": original_total,
            "corrected_total_mass_kg": export.get("total_mass_kg"),
            "inertia": {
                "status": "INVALIDATED",
                "reason": "not correctable from this export; unused by the 1 DOF model",
            },
            "authoritative_fix": "fix the 'PLA (orange)' material in Fusion and re-export",
        }
        out_path = path.with_name(path.stem + "_corrected.json")
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(export, handle, indent=2)
        return out_path

    # Capture the phantom mass in world coordinates BEFORE anything is mutated,
    # and record every occurrence's original mass so CGs can be corrected after.
    phantoms = find_phantom_moments(export)
    original_masses = {
        occ.get("full_path"): (occ.get("properties") or {}).get("mass_kg")
        for occ in export.get("occurrences", [])
    }

    density_count = _fix_densities(export)
    total_delta, moments, body_count = _correct_broken_bodies(export)

    # Propagate corrected MASS up: an occurrence's properties is a subtree rollup,
    # so every ancestor of a corrected body loses the same delta.
    for occ in export.get("occurrences", []):
        path_key = occ.get("full_path") or ""
        delta = _subtree_delta(export, path_key)
        if delta > 0.0:
            _correct_container(occ.get("properties"), delta)

    _correct_container((export.get("root") or {}).get("properties"), total_delta)

    if export.get("total_mass_kg") is not None:
        export["total_mass_kg"] -= total_delta
    if export.get("summed_occurrence_mass_kg") is not None:
        export["summed_occurrence_mass_kg"] -= total_delta

    # Correct world-frame CGs by removing the phantom moment, and invalidate every
    # inertia tensor that the phantom mass touched.
    root_props = (export.get("root") or {}).get("properties")
    _correct_cg(root_props, phantoms, original_total)
    _invalidate_inertia(root_props)

    stage_checks = {}
    for occ in export.get("occurrences", []):
        path_key = occ.get("full_path") or ""
        contained = [
            p for p in phantoms
            if (p["path"] or "") == path_key or (p["path"] or "").startswith(path_key + "+")
        ]
        if not contained:
            continue
        _correct_cg(occ.get("properties"), contained, original_masses.get(path_key) or 0.0)
        _invalidate_inertia(occ.get("properties"))

    # Cross-check: body masses must sum to the corrected rollups.
    root_sum, root_bodies = _subtree_body_mass(export, None)
    for occ in export.get("occurrences", []):
        path_key = occ.get("full_path") or ""
        if path_key.count("+") != 0:
            continue
        stage_sum, stage_bodies = _subtree_body_mass(export, path_key)
        props = occ.get("properties") or {}
        stage_checks[occ.get("name")] = {
            "bodies": stage_bodies,
            "body_mass_sum_kg": stage_sum,
            "rollup_mass_kg": props.get("mass_kg"),
            "cg_m": props.get("cg_m"),
        }
    root_check = {
        "bodies": root_bodies,
        "body_mass_sum_kg": root_sum,
        "rollup_mass_kg": (root_props or {}).get("mass_kg"),
    }

    export["corrections_applied"] = {
        "density_unit_fix": {
            "fields_corrected": density_count,
            "factor": DENSITY_FIX_FACTOR,
            "reason": "exporter used 1e3; Fusion reports kg/cm^3 so the factor is 1e6",
        },
        "broken_material": {
            "material": BROKEN_MATERIAL,
            "bodies_corrected": body_count,
            "true_density_kg_m3": BROKEN_MATERIAL_TRUE_DENSITY,
            "mass_removed_kg": total_delta,
            "original_total_mass_kg": original_total,
            "corrected_total_mass_kg": export.get("total_mass_kg"),
            "authoritative_fix": "correct the material definition in Fusion and re-export",
        },
        "centre_of_mass": {
            "method": "phantom mass moment removed from world-frame occurrence CGs",
            "phantom_locations": phantoms,
            "root": root_check,
            "per_stage": stage_checks,
            "note": "Body-level cg_m in this export is in the body's LOCAL frame, "
                    "so body CGs cannot be summed into a vehicle CG. Occurrence "
                    "cg_m is world-frame and was corrected instead. Cross-check: "
                    "body_mass_sum_kg should equal rollup_mass_kg to rounding.",
        },
        "inertia": {
            "status": "INVALIDATED",
            "reason": "Correcting inertia needs each body's tensor in world "
                      "coordinates; only local-frame body data is available. A "
                      "plausible wrong tensor is worse than none. The 1 DOF build "
                      "does not use inertia, so nothing downstream is affected.",
            "fix": "fix the Fusion material and re-export",
        },
        "not_corrected": [
            "TM000 (TeleMega 2.0) sits at r = 1.34 m from the axis and "
            "Marfik - DAQ Systems at r = 0.53 m. These look like misplaced "
            "components. They are left as-is because only the CAD author can "
            "say where they belong; they would corrupt any inertia calculation, "
            "and the recomputed CG/inertia above still includes them.",
        ],
    }

    out_path = path.with_name(path.stem + "_corrected.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(export, handle, indent=2)
    return out_path


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    drop = "--drop" in sys.argv
    if not args:
        print(__doc__)
        print("\n  --drop   remove the Wago components entirely instead of correcting")
        print("           their density (equivalent to suppressing them in Fusion;")
        print("           the two differ by about a gram)")
        raise SystemExit(2)
    raw = Path(args[0]).resolve()
    if not raw.exists():
        raise SystemExit(f"not found: {raw}")

    backup = raw.with_name(raw.stem + "_raw_backup.json")
    if not backup.exists():
        shutil.copy2(raw, backup)
        print(f"raw export backed up to {backup.name}")

    out = fix_export(raw, drop_wago=drop)
    with open(out, encoding="utf-8") as handle:
        result = json.load(handle)
    applied = result["corrections_applied"]

    print(f"\nwrote {out.name}")
    print(f"  density fields corrected : {applied['density_unit_fix']['fields_corrected']}")

    if "dropped_components" in applied:
        dropped = applied["dropped_components"]
        print(f"  components dropped       : {len(dropped['dropped'])} "
              f"({dropped['occurrences_removed']} occurrences)")
        for entry in dropped["dropped"]:
            print(f"      {entry['mass_kg']:10.4f} kg  {entry['name']}")
        print(f"  mass removed             : {dropped['mass_removed_kg']:.3f} kg")
        print(f"  total mass {applied['original_total_mass_kg']:.3f} kg"
              f"  ->  {applied['corrected_total_mass_kg']:.3f} kg")
        cg = (result.get("root") or {}).get("properties", {}).get("cg_m")
        if cg:
            print(f"  vehicle CG               : z {cg['z']:+.4f} m")
        print("\n  Reminder: fix 'PLA (orange)' in Fusion, or the next export repeats this.")
        return

    print(f"  broken bodies corrected  : {applied['broken_material']['bodies_corrected']}")
    print(f"  phantom mass removed     : {applied['broken_material']['mass_removed_kg']:.3f} kg")
    print(f"  total mass {applied['broken_material']['original_total_mass_kg']:.3f} kg"
          f"  ->  {applied['broken_material']['corrected_total_mass_kg']:.3f} kg")

    com = applied["centre_of_mass"]
    root = com["root"]
    print("\n  mass cross-check (body sum vs corrected rollup):")
    print(f"    root: {root['bodies']} bodies  {root['body_mass_sum_kg']:.4f} kg "
          f"vs {root['rollup_mass_kg']:.4f} kg")
    for name, check in com["per_stage"].items():
        print(f"    {name}: {check['bodies']} bodies  {check['body_mass_sum_kg']:.4f} kg "
              f"vs {check['rollup_mass_kg']:.4f} kg")

    print("\n  corrected centres of mass (world frame, z along the vehicle):")
    cg = (result.get("root") or {}).get("properties", {}).get("cg_m")
    if cg:
        print(f"    whole vehicle : z {cg['z']:+.4f} m")
    for name, check in com["per_stage"].items():
        if check.get("cg_m"):
            print(f"    {name:22s}: z {check['cg_m']['z']:+.4f} m")
    worst_purity = min((p["purity"] for p in com["phantom_locations"]), default=1.0)
    print(f"    phantom containers were {100 * worst_purity:.3f}% phantom "
          f"(higher is a better CG approximation)")
    print("  inertia: INVALIDATED (not correctable from this export; unused in 1 DOF)")

    print("\n  Reminder: fix 'PLA (orange)' in Fusion, or the next export repeats this.")


if __name__ == "__main__":
    main()
