#!/usr/bin/env python3
"""The CAD bench: 70 parts, an exact B-rep oracle, and gates that are allowed to fail.

The bench exists to expose where the field representation is WEAKER than the exact kernel beside it,
not to produce a score. There is deliberately no total. Each part is built as an exact solid, exported
to STEP and re-imported, rasterised into the signed field, and then the same three quantities --
volume, surface area, centre of mass -- are read off both representations and compared. A fourth
family runs the field back out through the section-loft recipe into a new solid and compares that
against the original. A fifth measures the distance field against the exact surface distance, and a
sixth runs the subtractive manufacturing gates.

The catalogue is 70 parts: 10 hand fixtures (box, rotated box, cylinder, rotated cylinder, sphere,
holed plate, bracket, recipe hole, thin wall, undercut) and 60 parameter parts drawn in three rounds
of 20 from 60 candidates by an exact fixed-cardinality determinantal sample over five varied
parameters (wall 0.15-3.15 mm, corner radius 2-6 mm, undercut 0-0.8 x wall, hole slenderness
h/(2r) 0.25-2, scale 0.5-1.5) at a 1 mm pitch.

WHAT THE MEASURED RUN SAYS, and it is not flattering to the field:

    volume, area and centre of mass all inside tolerance   11 / 70
      volume alone (voxel occupancy count)                 13 / 70
      area alone (marching-cubes isosurface)               27 / 70
      centre of mass alone                                 67 / 70
    distance field within 1.5 pitch of the exact distance  67 / 70
    STEP -> field -> loft -> STEP, all three gates         13 / 70
    turning, tool-nose radius reached                       4 / 4
    milling and drilling, zero voxel mismatch               2 / 2

    8 817 / 8 817 hash comparisons equal over two processes

The volume integral and the loft round trip are where the field loses, and that is the bench's point.
The centre of mass survives because it is a first moment of the same occupancy whose zeroth moment is
wrong: a systematically offset boundary cancels in the mean and does not cancel in the sum.

THE DETERMINISM NUMBER, precisely. 8 817 = 8 545 + 264 + 8. The 8 545 are the geometry half, compared
across two FRESH worker processes: every array hashed as its raw contiguous bytes (dtype and shape
recorded separately), every scalar in a part's metrics hashed individually as little-endian float64 or
int64, and one canonical-JSON hash per metrics record -- 4 868 from the parts, 2 340 from re-running
every screening candidate in its own process, 1 284 from the candidate generation, 53 from the
manufacturing rules. The 264 are the physics half's scalars, and the last 8 are six array hashes plus
two metrics hashes. Timings and STEP container bytes are excluded by construction; they are not
properties of the computation.

THE DENSITY-RATIO GATE IS NOT A GEOMETRY GATE. The bench carries a stiffness leg that reads
15 / 15 with a maximum relative deviation of 0.046 31242218, and an audit established exactly what
that number is worth:

  * every part is evaluated on THE SAME rectangular cantilever mesh. Only a single homogeneous
    density fraction is taken from the part's parameters; hole geometry, outline and rotation never
    reach the mesh.
  * the reference and the candidate differ only by rho_std = 0.985 x rho_ref, so the measured
    deviation follows the SIMP law exactly: at rho = 0.876238617139 the gate measures
    0.046 31242218 and the closed form predicts 0.046 31242218, to every digit printed.
  * HALVING THE PITCH CHANGES NOTHING. Re-running the whole leg at half pitch leaves all 250 numeric
    fields identical, all four array hashes identical, and the same metrics hash
    7a00ea2750abc7ab36081ae261f1e7ef357d83e79e4b69aa019218b77d9572ae both ways.
  * halving the FEM element size moves the reference by 17.412 984 %, and the coarse mesh deviates
    14.830 544 % from the fine one -- above the gate's own 5 % band, so "converged reference" is not
    established either.

It is therefore shipped here as what it is, `densitetskvotsgrind`: a check that the stiffness solver
reproduces the SIMP density ratio. Geometric validation of stiffness is ABSENT from this bench, and
the list of what the bench cannot measure (transient contact, plasticity, five-axis collision,
thermal, flow, continuous Hausdorff, tolerance stack-up) is longer than what it can.

The gradient legs read 15 / 15 at a step of 1e-5 with a maximum relative error of 1.74e-10 (contact)
and 3.87e-07 (stiffness). Those are single-component checks -- one shape parameter, one element -- and
a step sweep puts the stiffness leg's best value at 1e-4 (5.0e-08), not at 1e-5.

RUNNING IT. The full 70-part measurement needs the exact B-rep kernel and takes about five minutes
per process, so the per-part records of a measured run are carried here as data and the gate counts
are RECOMPUTED from the raw measurements by the gate functions below -- the count is never written
down twice. `python cadbank_v1.py` re-derives every count and the determinism total from the carried
records and prints them.
"""
import json
import os

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "cadbank_v1")

# The gate bands, exactly as the measured run applied them. They are relative except where a length
# is the honest unit: the centre of mass is compared in millimetres against one pitch, the distance
# field against 1.5 pitch, and the sampled Hausdorff against two pitches.
VOLYM_BAND = 0.05
YTA_BAND = 0.10
MASSCENTRUM_BAND_PITCH = 1.0
EDT_BAND_PITCH = 1.5
HAUSDORFF_BAND_PITCH = 2.0
RUNDRESA_VOLYM_BAND = 0.05
SEKTIONSINTEGRAL_BAND = 1e-7


def las_delar(sokvag=None):
    """The per-part records of a measured run: one dict per part, 70 of them."""
    with open(os.path.join(sokvag or DATA, "parts.json")) as fh:
        return json.load(fh)["parts"]


def _rel(a, b):
    return abs(a - b) / abs(b)


def grind_volym(del_):
    """Volume of the occupancy against the exact solid. The occupancy count is the shipped meter."""
    return _rel(del_["sdf_voxel_volume"], del_["brep_volume"]) <= VOLYM_BAND


def grind_yta(del_):
    """Surface area of the marching-cubes isosurface against the exact solid."""
    return _rel(del_["sdf_area"], del_["brep_area"]) <= YTA_BAND


def grind_masscentrum(del_):
    """First moment of the occupancy against the exact solid, in millimetres against one pitch."""
    e = del_["sdf_com_error_mm"]
    return e is not None and e <= del_["pitch"] * MASSCENTRUM_BAND_PITCH


def grind_avstandsfalt(del_):
    """The distance field against the exact surface distance at 96 deterministic lattice probes."""
    e = del_["edt_max_dev_mm"]
    return e is not None and e <= del_["pitch"] * EDT_BAND_PITCH


def grind_rundresa_volym(del_):
    """STEP -> field -> section loft -> STEP: volume of the rebuilt solid against the original."""
    v = del_["roundtrip_volume"]
    return v is not None and _rel(v, del_["brep_volume"]) <= RUNDRESA_VOLYM_BAND


def grind_rundresa_yta(del_):
    """Sampled Hausdorff distance between the input surface and the rebuilt one, against two pitches."""
    h = del_["hausdorff_sampled_mm"]
    return h is not None and h <= del_["pitch"] * HAUSDORFF_BAND_PITCH


def grind_sektionsintegral(del_):
    """The recipe's own section integral against the volume of the solid it rebuilds."""
    v, i = del_["roundtrip_volume"], del_["roundtrip_integral"]
    return v is not None and i is not None and _rel(v, i) < SEKTIONSINTEGRAL_BAND


def grind_rundresa_giltig(del_):
    """The rebuilt solid is a valid body and every recipe op reported PASS."""
    return bool(del_["roundtrip_valid"])


FAMILJ_A = {"volym": grind_volym, "yta": grind_yta, "masscentrum": grind_masscentrum}
FAMILJ_B = {"avstandsfalt": grind_avstandsfalt}
FAMILJ_C = {"volym": grind_rundresa_volym, "yta": grind_rundresa_yta,
            "integral": grind_sektionsintegral, "giltig": grind_rundresa_giltig}


def familjeutfall(delar):
    """Re-derive every gate count from the raw measurements. No count is stored anywhere."""
    ut = {}
    for namn, familj in (("a", FAMILJ_A), ("b", FAMILJ_B), ("c", FAMILJ_C)):
        per = {g: sum(1 for d in delar if f(d)) for g, f in familj.items()}
        per["samtidigt"] = sum(1 for d in delar if all(f(d) for f in familj.values()))
        ut[namn] = per
    ut["n"] = len(delar)
    return ut


def tillverkningsutfall(sokvag=None):
    """The subtractive gates: turning against the tool nose radius, milling and drilling against an
    independent voxel oracle. Turning passes when the measured root radius is within 0.05 mm of the
    geometric one; milling and drilling require zero mismatched voxels."""
    with open(os.path.join(sokvag or DATA, "tillverkning.json")) as fh:
        t = json.load(fh)
    svarv = [r for r in t["turning"] if abs(r["error_mm"]) <= t["turning_tolerance_mm"]]
    frasborr = [r for r in t["milling"] if r["mismatch_voxels"] <= t["milling_tolerance_voxels"]]
    return {"svarv": (len(svarv), len(t["turning"])),
            "fras_borr": (len(frasborr), len(t["milling"])),
            "max_svarvfel_mm": max(abs(r["error_mm"]) for r in t["turning"]),
            "restmaterial_kort_borr_voxlar": t["short_drill_residual_voxels"]}


def jamfor_hashposter(a, b):
    """The determinism comparator: two hash records in, (n_jamforelser, n_lika) out.

    A record is {"arrays": {name: {"sha256": ...}}, "numbers": {name: sha}, "metrics_sha256": sha}.
    Every array, every individual scalar and the canonical metrics hash count as one comparison each.
    A name present in one record and not the other is a comparison that is NOT equal -- a dropped
    field must never pass as agreement."""
    n = lika = 0
    for nyckel in ("arrays", "numbers"):
        for namn in sorted(set(a.get(nyckel, {})) | set(b.get(nyckel, {}))):
            n += 1
            va, vb = a.get(nyckel, {}).get(namn), b.get(nyckel, {}).get(namn)
            if isinstance(va, dict):
                va = va.get("sha256")
            if isinstance(vb, dict):
                vb = vb.get("sha256")
            lika += int(va is not None and va == vb)
    if "metrics_sha256" in a or "metrics_sha256" in b:
        n += 1
        lika += int(a.get("metrics_sha256") is not None
                    and a.get("metrics_sha256") == b.get("metrics_sha256"))
    return n, lika


def determinismutfall(sokvag=None):
    """Total the carried comparison ledger: geometry groups, physics scalars, and the extra hashes.

    The extra hashes are re-compared here from their two recorded digests, so the last eight of the
    8 817 are decided by this function and not by a stored count."""
    with open(os.path.join(sokvag or DATA, "determinism.json")) as fh:
        d = json.load(fh)
    n = sum(g["comparisons"] for g in d["part_a_groups"].values()) + d["part_b_numbers"]["comparisons"]
    lika = sum(g["equal"] for g in d["part_a_groups"].values()) + d["part_b_numbers"]["equal"]
    for h in d["extra_hashes"]:
        n += 1
        lika += int(h["run1"] == h["run2"])
    return {"jamforelser": n, "lika": lika, "grupper": d["part_a_groups"],
            "rapporterat_i_korningen": d["rapporterat_i_korningen"]}


def densitetskvotsgrind(sokvag=None):
    """The stiffness leg, under the name the audit left it with.

    Returns the measured maximum relative deviation, the SIMP prediction for the same density ratio,
    and the pitch control that shows the leg is pitch-blind."""
    with open(os.path.join(sokvag or DATA, "densitetsgrind.json")) as fh:
        d = json.load(fh)
    pk = d["pitch_control"]
    return {"n_delar": d["summary"]["n_parts"],
            "stelhet_passerade": d["summary"]["u14_compliance_passed"],
            "max_relativ_avvikelse": d["summary"]["max_u14_comp_rel_err"],
            "kontaktmarginal_passerade": d["summary"]["u11_margin_passed"],
            "max_kontaktmarginalfel": d["summary"]["max_u11_margin_rel_err"],
            "gradient_passerade": d["summary"]["u14_grad_passed"],
            "halverad_pitch_identisk": (pk["metrics_sha256_original"] == pk["metrics_sha256_half_pitch"]
                                        and pk["numeric_equal"] == pk["numeric_fields"]),
            "halverad_pitch_numeriska_falt": pk["numeric_fields"],
            "geometrisk_validering": False}


def simp_forhallande(rho, faktor=0.985, e0=1000.0, rho_min=1e-3, p=3.0):
    """The closed form the stiffness gate actually measures: the compliance ratio between a uniform
    density rho and the same field scaled by `faktor`, under SIMP interpolation
    E(rho) = E_min + (E0 - E_min) rho^p with E_min = rho_min E0. No geometry enters.

    At rho = 0.5 this gives 0.045 999 078 409 and at rho = 0.876 238 617 139 it gives
    0.046 312 422 180, which are the two deviations the gate measured, on both the 40x20 and the
    80x40 mesh."""
    e_min = rho_min * e0
    return (e_min + (e0 - e_min) * rho ** p) / (e_min + (e0 - e_min) * (faktor * rho) ** p) - 1.0


def _selftest():
    delar = las_delar()
    fam = familjeutfall(delar)
    till = tillverkningsutfall()
    det = determinismutfall()
    dens = densitetskvotsgrind()
    print(f"parts {fam['n']}")
    print(f"a volume {fam['a']['volym']}/{fam['n']}  area {fam['a']['yta']}/{fam['n']}  "
          f"com {fam['a']['masscentrum']}/{fam['n']}  simultaneous {fam['a']['samtidigt']}/{fam['n']}")
    print(f"b distance field {fam['b']['avstandsfalt']}/{fam['n']}")
    print(f"c round trip simultaneous {fam['c']['samtidigt']}/{fam['n']}")
    print(f"turning {till['svarv'][0]}/{till['svarv'][1]}  "
          f"mill+drill {till['fras_borr'][0]}/{till['fras_borr'][1]}")
    print(f"determinism {det['lika']}/{det['jamforelser']}")
    print(f"density-ratio gate {dens['stelhet_passerade']}/{dens['n_delar']} "
          f"max {dens['max_relativ_avvikelse']:.11f}  pitch-blind={dens['halverad_pitch_identisk']}")
    fel = []
    if fam["n"] != 70:
        fel.append("catalogue is not 70 parts")
    if det["lika"] != det["jamforelser"]:
        fel.append("determinism ledger has an unequal comparison")
    if dens["geometrisk_validering"]:
        fel.append("the density-ratio gate must not claim geometric validation")
    if fel:
        raise SystemExit("FAIL: " + "; ".join(fel))
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
