#!/usr/bin/env python3
"""
RNP Bench backend — real all-atom molecular dynamics with OpenMM.

The browser front end (rnp-bench.html) works on its own for preparation and
analysis. Start this server and it also gains a Run tab that actually
integrates equations of motion.

    pip install "openmm>=8.1" pdbfixer fastapi "uvicorn[standard]" numpy
    python rnp_server.py

Then open http://localhost:8000/ in a browser.

Everything stays on this machine. No structure or trajectory leaves it.
"""

from __future__ import annotations

import argparse
import io
import zipfile
import secrets
import shutil
import json
import math
import os
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field

import openmm as mm
import openmm.app as app
from openmm import unit
from pdbfixer import PDBFixer

VERSION = "2026.09.12-2"
KB_KJ = 0.008314462618  # kJ/mol/K
HERE = os.path.dirname(os.path.abspath(__file__))


class Config:
    """Set from the command line in main(). Defaults are the safe localhost case."""
    token: str = ""
    host: str = "127.0.0.1"
    origins: list = []
    devices: list = ["0"]          # one concurrent job per GPU listed
    max_atoms: int = 400_000
    max_series_points: int = 4000
    work_dir: str = os.path.join(HERE, "rnp_runs")

    def local_only(self) -> bool:
        return self.host in ("127.0.0.1", "localhost", "::1")


CFG = Config()

# --------------------------------------------------------------------------
# Force fields. Names match the ones the front end offers, so a plan built in
# the browser and a run started here use the same parameters.
# --------------------------------------------------------------------------
FFDEF: dict[str, dict[str, Any]] = {
    "amber-ff19sb-ol3-opc": {
        "label": "AMBER ff19SB / \u03c7OL3 / OPC",
        "xml": ["amber14/protein.ff14SB.xml", "amber14/RNA.OL3.xml",
                "amber14/DNA.OL15.xml", "amber14/opc.xml"],
        "water_geometry": "tip4pew",   # OPC uses the 4-site geometry template
        "cutoff_nm": 0.9,
        "note": ("OpenMM ships ff14SB rather than ff19SB. ff19SB needs the CMAP "
                 "variant that is not in the bundled amber14 set, so this run uses "
                 "ff14SB with OPC water. The generated GROMACS/AMBER inputs still "
                 "use ff19SB — they run in AmberTools, which has it."),
    },
    "amber-ff14sb-ol3-tip3p": {
        "label": "AMBER ff14SB / \u03c7OL3 / TIP3P",
        "xml": ["amber14/protein.ff14SB.xml", "amber14/RNA.OL3.xml",
                "amber14/DNA.OL15.xml", "amber14/tip3p.xml"],
        "water_geometry": "tip3p",
        "cutoff_nm": 0.9,
        "note": "Exactly the combination the generated AMBER inputs describe.",
    },
    "amber-ff99sbildn-ol3-tip3p": {
        "label": "AMBER ff99SB-ILDN / \u03c7OL3 / TIP3P",
        "xml": ["amber99sbildn.xml", "amber14/RNA.OL3.xml", "tip3p.xml"],
        "water_geometry": "tip3p",
        "cutoff_nm": 0.9,
        "note": ("Legacy protein parameters combined with the modern \u03c7OL3 RNA "
                 "correction. Only for reproducing older work."),
    },
    "charmm36m": {
        "label": "CHARMM36m / CHARMM36 nucleic / TIP3P",
        "xml": ["charmm36.xml", "charmm36/water.xml"],
        "water_geometry": "tip3p",
        "cutoff_nm": 1.2,
        "note": ("CHARMM systems are normally built with CHARMM-GUI. The bundled "
                 "charmm36.xml covers standard residues only; modified nucleotides "
                 "will fail to match a template."),
    },
}

WATER_GEOMETRY = {"opc": "tip4pew", "tip3p": "tip3p", "spce": "spce", "tip4pew": "tip4pew"}
BOX_SHAPE = {"dodecahedron": "dodecahedron", "octahedron": "octahedron", "cubic": "cube"}


# --------------------------------------------------------------------------
# Request models
# --------------------------------------------------------------------------
class Settings(BaseModel):
    ff: str = "amber-ff14sb-ol3-tip3p"
    water: str = "tip3p"
    shape: str = "dodecahedron"
    padding_A: float = 12.0
    salt_M: float = 0.15
    cation: str = "K"
    temp_K: float = 300.0
    press_bar: float = 1.0
    dt_fs: float = 2.0
    ph: float = 7.0
    keep_crystal_water: bool = False
    minimize_steps: int = 2000
    heat_ps: float = 100.0
    equil_ps: float = 500.0
    prod_ps: float = 2000.0
    save_ps: float = 10.0
    report_ps: float = 1.0
    platform: str = "auto"
    seed: int = 0

    def ffdef(self) -> dict[str, Any]:
        if self.ff not in FFDEF:
            raise HTTPException(400, f"unknown force field {self.ff}")
        return FFDEF[self.ff]


class PrepareRequest(BaseModel):
    pdb: str
    settings: Settings = Field(default_factory=Settings)


class RunRequest(BaseModel):
    pdb: str
    name: str = "rnp"
    settings: Settings = Field(default_factory=Settings)


# --------------------------------------------------------------------------
# Job state
# --------------------------------------------------------------------------
@dataclass
class Job:
    id: str
    name: str
    settings: dict
    device: str = "0"
    work_dir: str = ""
    status: str = "queued"          # queued running done failed stopped
    stage: str = "waiting"
    message: str = ""
    created: float = field(default_factory=time.time)
    progress: float = 0.0
    system: dict = field(default_factory=dict)
    series: dict = field(default_factory=lambda: {
        "t_ps": [], "pe": [], "ke": [], "temp": [], "volume": [], "density": [],
        "stage": [], "restraint_k": [],
    })
    log: list = field(default_factory=list)
    series_decimated: int = 0
    energy: dict = field(default_factory=lambda: {"status": "none", "progress": 0.0,
                                                  "series": None, "error": "", "note": ""})
    frames: list = field(default_factory=list)   # solute-only coords, Angstrom
    frame_times: list = field(default_factory=list)
    solute_pdb_header: str = ""
    solute_topology: Any = None
    speed_ns_day: float = 0.0
    stop_flag: bool = False
    error: str = ""

    def note(self, text: str) -> None:
        self.log.append({"t": time.time() - self.created, "text": text})
        self.message = text
        print(f"[{self.id[:8]}] {text}", flush=True)

    def public(self, series_from: int = 0) -> dict:
        return {
            "id": self.id, "name": self.name, "status": self.status, "stage": self.stage,
            "message": self.message, "progress": round(self.progress, 4),
            "system": self.system, "speed_ns_day": round(self.speed_ns_day, 3),
            "n_frames": len(self.frames), "error": self.error,
            "elapsed_s": round(time.time() - self.created, 1),
            "series_from": series_from,
            "series_decimated": self.series_decimated,
            "series": {k: v[series_from:] for k, v in self.series.items()},
            "log": self.log[-40:],
        }


JOBS: dict[str, Job] = {}
JOB_LOCK = threading.Lock()


# --------------------------------------------------------------------------
# System building
# --------------------------------------------------------------------------
def pick_platform(name: str):
    avail = {mm.Platform.getPlatform(i).getName(): mm.Platform.getPlatform(i)
             for i in range(mm.Platform.getNumPlatforms())}
    if name != "auto" and name in avail:
        return avail[name]
    for pref in ("CUDA", "HIP", "OpenCL", "CPU", "Reference"):
        if pref in avail:
            return avail[pref]
    raise RuntimeError("no OpenMM platform available")


def build_system(pdb_text: str, s: Settings, note=print):
    """Clean the structure, solvate it, and build an OpenMM system."""
    ff_def = s.ffdef()

    if not any(l.startswith(("ATOM", "HETATM")) for l in pdb_text.splitlines()):
        raise ValueError("no ATOM or HETATM records found — is this really a PDB file?")

    note("reading structure and adding missing atoms")
    try:
        fixer = PDBFixer(pdbfile=io.StringIO(pdb_text))
    except Exception as exc:
        raise ValueError(f"could not parse the structure: {exc}") from exc
    fixer.findMissingResidues()
    fixer.missingResidues = {}                  # do not model absent loops
    fixer.findNonstandardResidues()
    nonstd = list(fixer.nonstandardResidues)
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(s.keep_crystal_water)
    fixer.findMissingAtoms()
    n_missing = sum(len(v) for v in fixer.missingAtoms.values())
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(s.ph)

    n_solute = fixer.topology.getNumAtoms()
    n_h = sum(1 for a in fixer.topology.atoms()
              if a.element is not None and a.element.symbol == "H")
    note(f"solute {n_solute} atoms ({n_h} hydrogens) at pH {s.ph}")

    forcefield = app.ForceField(*ff_def["xml"])
    modeller = app.Modeller(fixer.topology, fixer.positions)
    modeller.addExtraParticles(forcefield)

    geometry = WATER_GEOMETRY.get(s.water, ff_def["water_geometry"])
    note(f"solvating: {s.shape}, {s.padding_A:.1f} A padding, {s.salt_M} M {s.cation}Cl")
    modeller.addSolvent(
        forcefield,
        model=geometry,
        padding=s.padding_A / 10.0 * unit.nanometer,
        boxShape=BOX_SHAPE.get(s.shape, "dodecahedron"),
        ionicStrength=s.salt_M * unit.molar,
        positiveIon=f"{s.cation}+",
        negativeIon="Cl-",
        neutralize=True,
    )

    hydrogen_mass = 4.0 if s.dt_fs >= 4 else 1.5
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=ff_def["cutoff_nm"] * unit.nanometer,
        constraints=app.HBonds,
        rigidWater=True,
        hydrogenMass=hydrogen_mass * unit.amu,
    )

    counts = {"water": 0, "ion": 0, "protein": 0, "nucleic": 0, "other": 0}
    ion_names = {"NA", "K", "CL", "MG", "CA", "ZN", "SOD", "POT", "CLA"}
    for r in modeller.topology.residues():
        n = r.name.upper()
        if n in ("HOH", "WAT"):
            counts["water"] += 1
        elif n in ion_names:
            counts["ion"] += 1
        elif n in app.PDBFile._standardResidues[:20] or len(n) == 3:
            counts["protein"] += 1
        elif n in ("A", "U", "G", "C", "DA", "DT", "DG", "DC"):
            counts["nucleic"] += 1
        else:
            counts["other"] += 1

    charge = 0.0
    for f in system.getForces():
        if isinstance(f, mm.NonbondedForce):
            charge = sum(f.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
                         for i in range(f.getNumParticles()))
            break

    box = modeller.topology.getUnitCellDimensions()
    summary = {
        "total_atoms": modeller.topology.getNumAtoms(),
        "solute_atoms": n_solute,
        "waters": counts["water"],
        "ions": counts["ion"],
        "residues": modeller.topology.getNumResidues(),
        "net_charge_e": round(charge, 3),
        "box_nm": [round(float(box[i].value_in_unit(unit.nanometer)), 3) for i in range(3)],
        "constraints": system.getNumConstraints(),
        "force_field": ff_def["label"],
        "water_model": s.water,
        "hydrogen_mass_amu": hydrogen_mass,
        "missing_atoms_added": n_missing,
        "solute_hydrogens": n_h,
        "nonstandard_replaced": [str(x) for x in nonstd],
        "ff_note": ff_def.get("note", ""),
    }
    if summary["total_atoms"] > CFG.max_atoms:
        raise ValueError(
            f"{summary['total_atoms']:,} atoms exceeds this server's limit of "
            f"{CFG.max_atoms:,}. Reduce the padding, drop chains you do not need, "
            f"or restart the server with a larger --max-atoms.")
    return forcefield, modeller, system, summary


def solute_indices(topology) -> list[int]:
    skip = {"HOH", "WAT"}
    ions = {"NA", "K", "CL", "SOD", "POT", "CLA"}
    out = []
    for r in topology.residues():
        n = r.name.upper()
        if n in skip or n in ions:
            continue
        for a in r.atoms():
            if a.element is not None:
                out.append(a.index)
    return out


def solute_subset(topology, positions, idx: list[int]):
    """The solute on its own: a topology plus a PDB header. Frames are appended as
    MODEL blocks, and keeping the topology lets us write a real DCD as well."""
    sub = app.Modeller(topology, positions)
    keep = set(idx)
    sub.delete([a for a in sub.topology.atoms() if a.index not in keep])
    buf = io.StringIO()
    app.PDBFile.writeFile(sub.topology, sub.positions, buf, keepIds=True)
    return sub.topology, sub.positions, buf.getvalue()


# --------------------------------------------------------------------------
# The run itself
# --------------------------------------------------------------------------
def run_job(job: Job, pdb_text: str, s: Settings) -> None:
    try:
        job.status = "running"
        job.stage = "prepare"
        forcefield, modeller, system, summary = build_system(pdb_text, s, job.note)
        job.system = summary
        if summary["ff_note"]:
            job.note(summary["ff_note"])

        idx = solute_indices(modeller.topology)
        job.solute_topology, _sub_pos, job.solute_pdb_header = solute_subset(
            modeller.topology, modeller.positions, idx)

        # everything needed to rebuild this run later
        if job.work_dir:
            art = lambda n: os.path.join(job.work_dir, n)
            with open(art("input.pdb"), "w") as f:
                f.write(pdb_text)
            with open(art("system_solvated.pdb"), "w") as f:
                app.PDBFile.writeFile(modeller.topology, modeller.positions, f, keepIds=True)
            with open(art("topology_solute.pdb"), "w") as f:
                f.write(job.solute_pdb_header)
            with open(art("settings.json"), "w") as f:
                json.dump(job.settings, f, indent=2)

        # positional restraints, released in stages during equilibration
        restraint = mm.CustomExternalForce("k*periodicdistance(x,y,z,x0,y0,z0)^2")
        restraint.addGlobalParameter("k", 0.0)
        for p in ("x0", "y0", "z0"):
            restraint.addPerParticleParameter(p)
        heavy = [i for i in idx if modeller.topology.getNumAtoms() and
                 list(modeller.topology.atoms())[i].element.symbol != "H"]
        pos_nm = modeller.positions.value_in_unit(unit.nanometer)
        for i in heavy:
            restraint.addParticle(i, [pos_nm[i][0], pos_nm[i][1], pos_nm[i][2]])
        system.addForce(restraint)
        system.addForce(mm.MonteCarloBarostat(s.press_bar * unit.bar, s.temp_K * unit.kelvin, 25))

        dt = s.dt_fs * unit.femtoseconds
        integrator = mm.LangevinMiddleIntegrator(
            s.temp_K * unit.kelvin, 1.0 / unit.picosecond, dt)
        if s.seed:
            integrator.setRandomNumberSeed(int(s.seed))
        platform = pick_platform(s.platform)
        props = {}
        if platform.getName() in ("CUDA", "HIP", "OpenCL"):
            props = {"DeviceIndex": job.device, "Precision": "mixed"}
        sim = app.Simulation(modeller.topology, system, integrator, platform, props)
        sim.context.setPositions(modeller.positions)
        job.system["platform"] = platform.getName()
        job.system["device"] = job.device
        job.note(f"platform {platform.getName()}, {summary['total_atoms']} atoms, "
                 f"{s.dt_fs} fs step")

        if job.work_dir:
            # The System XML is the force field exactly as applied: every charge and
            # every bonded term. With the solvated PDB it reproduces this run without
            # needing the same force-field files installed.
            with open(os.path.join(job.work_dir, "system.xml"), "w") as f:
                f.write(mm.XmlSerializer.serialize(system))
            with open(os.path.join(job.work_dir, "integrator.xml"), "w") as f:
                f.write(mm.XmlSerializer.serialize(integrator))

        dof = 3 * system.getNumParticles() - system.getNumConstraints() - 3
        total_mass = sum(system.getParticleMass(i).value_in_unit(unit.dalton)
                         for i in range(system.getNumParticles()))

        report_every = max(1, int(round(s.report_ps * 1000 / s.dt_fs)))
        save_every = max(1, int(round(s.save_ps * 1000 / s.dt_fs)))
        clock = {"t_ps": 0.0}

        def record(stage: str, k_value: float) -> None:
            st = sim.context.getState(getEnergy=True, getPositions=False)
            pe = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            ke = st.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
            vol = st.getPeriodicBoxVolume().value_in_unit(unit.nanometer**3)
            temp = 2 * ke / (dof * KB_KJ)
            dens = total_mass / (vol * 602.2140762)  # amu/nm^3 -> g/cm^3
            ser = job.series
            ser["t_ps"].append(round(clock["t_ps"], 4))
            ser["pe"].append(round(pe, 2))
            ser["ke"].append(round(ke, 2))
            ser["temp"].append(round(temp, 2))
            ser["volume"].append(round(vol, 3))
            ser["density"].append(round(dens, 4))
            ser["stage"].append(stage)
            ser["restraint_k"].append(k_value)
            # a long run would otherwise accumulate hundreds of thousands of points
            if len(ser["t_ps"]) > CFG.max_series_points * 2:
                for key in ser:
                    ser[key] = ser[key][::2]
                job.series_decimated += 1

        def save_frame() -> None:
            st = sim.context.getState(getPositions=True, enforcePeriodicBox=False)
            p = st.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
            job.frames.append(np.asarray(p[idx], dtype=np.float32))
            job.frame_times.append(round(clock["t_ps"], 3))
            # spill to disk so a long run does not live entirely in RAM
            if job.work_dir and len(job.frames) % 200 == 0:
                np.save(os.path.join(job.work_dir, "frames.npy"),
                        np.asarray(job.frames, dtype=np.float32))

        def advance(stage: str, ps: float, k_value: float, save: bool,
                    span: tuple[float, float]) -> bool:
            """Run one stage in report-sized chunks so the browser sees it live."""
            nsteps = int(round(ps * 1000 / s.dt_fs))
            if nsteps <= 0:
                return True
            sim.context.setParameter("k", k_value)
            job.stage = stage
            done = 0
            t_wall = time.time()
            since_save = 0
            while done < nsteps:
                if job.stop_flag:
                    job.note(f"stopped during {stage}")
                    return False
                chunk = min(report_every, nsteps - done)
                sim.step(chunk)
                done += chunk
                since_save += chunk
                clock["t_ps"] += chunk * s.dt_fs / 1000.0
                record(stage, k_value)
                if save and since_save >= save_every:
                    save_frame()
                    since_save = 0
                frac = done / nsteps
                job.progress = span[0] + (span[1] - span[0]) * frac
                el = time.time() - t_wall
                if el > 1:
                    job.speed_ns_day = (done * s.dt_fs / 1e6) / (el / 86400.0)
            return True

        # ---- minimise -------------------------------------------------
        job.stage = "minimise"
        sim.context.setParameter("k", 1000.0)
        e0 = sim.context.getState(getEnergy=True).getPotentialEnergy()
        job.note("minimising with the solute restrained")
        sim.minimizeEnergy(maxIterations=s.minimize_steps)
        e1 = sim.context.getState(getEnergy=True).getPotentialEnergy()
        job.system["pe_before_kJ"] = round(e0.value_in_unit(unit.kilojoule_per_mole), 1)
        job.system["pe_after_kJ"] = round(e1.value_in_unit(unit.kilojoule_per_mole), 1)
        job.note(f"potential energy {job.system['pe_before_kJ']:,.0f} -> "
                 f"{job.system['pe_after_kJ']:,.0f} kJ/mol")
        job.progress = 0.05
        record("minimise", 1000.0)

        # ---- heat with restraints -------------------------------------
        if s.seed:
            sim.context.setVelocitiesToTemperature(s.temp_K * unit.kelvin, int(s.seed))
        else:
            sim.context.setVelocitiesToTemperature(s.temp_K * unit.kelvin)
        job.note(f"heating to {s.temp_K:.0f} K, restraints on")
        if not advance("heat", s.heat_ps, 1000.0, False, (0.05, 0.15)):
            job.status = "stopped"
            return

        # ---- release the restraints in stages --------------------------
        stages = [(500.0, 0.25), (100.0, 0.25), (10.0, 0.25), (0.0, 0.25)]
        lo = 0.15
        for k_value, frac in stages:
            ps = s.equil_ps * frac
            hi = lo + 0.30 * frac
            job.note(f"equilibrating {ps:g} ps at k = {k_value:.0f} kJ/mol/nm^2")
            if not advance("equilibrate", ps, k_value, False, (lo, hi)):
                job.status = "stopped"
                return
            lo = hi

        # ---- production ------------------------------------------------
        job.note(f"production {s.prod_ps:.0f} ps, saving every {s.save_ps:.0f} ps")
        save_frame()
        if not advance("produce", s.prod_ps, 0.0, True, (0.45, 1.0)):
            job.status = "stopped"
            return

        if job.work_dir:
            st = sim.context.getState(getPositions=True, getVelocities=True,
                                      enforcePeriodicBox=False)
            with open(os.path.join(job.work_dir, "final_state.xml"), "w") as f:
                f.write(mm.XmlSerializer.serialize(st))
            with open(os.path.join(job.work_dir, "final_frame.pdb"), "w") as f:
                app.PDBFile.writeFile(sim.topology, st.getPositions(), f, keepIds=True)
            try:
                write_dcd(job)
            except Exception as exc:
                job.note(f"could not write the DCD: {exc}")

        job.progress = 1.0
        job.stage = "done"
        job.status = "done"
        job.note(f"finished: {len(job.frames)} frames, {job.speed_ns_day:.2f} ns/day")

    except Exception as exc:  # surfaced to the browser rather than swallowed
        job.status = "failed"
        job.stage = "failed"
        job.error = f"{type(exc).__name__}: {exc}"
        job.note(job.error)
        traceback.print_exc()


# --------------------------------------------------------------------------
# HTTP API
# --------------------------------------------------------------------------
def require_token(request: Request) -> None:
    """Every endpoint except /health needs the token once one is set."""
    if not CFG.token:
        return
    supplied = request.headers.get("x-api-token") or request.query_params.get("token")
    if not supplied or not secrets.compare_digest(supplied, CFG.token):
        raise HTTPException(401, "missing or wrong API token")


api = FastAPI(title="RNP Bench backend", dependencies=[])
def install_cors() -> None:
    """A wildcard origin is only acceptable while the server is on localhost.
    Once it listens on a real interface the caller must name its origins."""
    origins = CFG.origins or (["*"] if CFG.local_only() else [])
    api.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["*"],
        allow_headers=["*", "x-api-token"],
    )


def gpu_names() -> list:
    """Ask each accelerated platform what hardware it can see."""
    out = []
    for i in range(mm.Platform.getNumPlatforms()):
        p = mm.Platform.getPlatform(i)
        if p.getName() not in ("CUDA", "HIP", "OpenCL"):
            continue
        try:
            system = mm.System()
            system.addParticle(1.0)
            ctx = mm.Context(system, mm.VerletIntegrator(0.001), p,
                             {"DeviceIndex": CFG.devices[0]})
            for key in ("DeviceName", "DeviceIndex"):
                try:
                    out.append(f"{p.getName()}: {p.getPropertyValue(ctx, key)}")
                    break
                except Exception:
                    continue
            del ctx
        except Exception as exc:
            out.append(f"{p.getName()}: present but unusable ({exc})")
    return out


@api.get("/health")
def health():
    plats = [mm.Platform.getPlatform(i).getName()
             for i in range(mm.Platform.getNumPlatforms())]
    fast = [p for p in plats if p in ("CUDA", "HIP", "OpenCL")]
    return {
        "ok": True,
        "version": VERSION,
        "openmm": mm.version.version,
        "platforms": plats,
        "accelerated": bool(fast),
        "recommended_platform": fast[0] if fast else "CPU",
        "gpus": gpu_names(),
        "slots": len(CFG.devices),
        "force_fields": {k: v["label"] for k, v in FFDEF.items()},
        "jobs": len(JOBS),
        "auth_required": bool(CFG.token),
        "max_atoms": CFG.max_atoms,
        "remote": not CFG.local_only(),
    }


@api.post("/prepare", dependencies=[Depends(require_token)])
def prepare(req: PrepareRequest):
    """Build the system and report what it would be, without running anything."""
    try:
        notes: list[str] = []
        _, _, _, summary = build_system(req.pdb, req.settings, notes.append)
        summary["log"] = notes
        return summary
    except Exception as exc:
        raise HTTPException(400, f"{type(exc).__name__}: {exc}")


@api.post("/run", dependencies=[Depends(require_token)])
def start(req: RunRequest):
    with JOB_LOCK:
        busy = [j for j in JOBS.values() if j.status == "running"]
        if len(busy) >= len(CFG.devices):
            raise HTTPException(409, f"all {len(CFG.devices)} GPU slot(s) busy "
                                     f"(running: {', '.join(j.id[:8] for j in busy)})")
        taken = {j.device for j in busy}
        device = next(d for d in CFG.devices if d not in taken)
        job = Job(id=uuid.uuid4().hex, name=req.name, device=device,
                  settings=json.loads(req.settings.model_dump_json()))
        job.work_dir = os.path.join(CFG.work_dir, job.id)
        os.makedirs(job.work_dir, exist_ok=True)
        JOBS[job.id] = job
    threading.Thread(target=run_job, args=(job, req.pdb, req.settings),
                     daemon=True).start()
    return {"id": job.id, "device": device}


@api.get("/jobs", dependencies=[Depends(require_token)])
def list_jobs():
    return [{"id": j.id, "name": j.name, "status": j.status, "stage": j.stage,
             "progress": round(j.progress, 3), "n_frames": len(j.frames),
             "created": j.created} for j in JOBS.values()]


@api.get("/jobs/{job_id}", dependencies=[Depends(require_token)])
def job_status(job_id: str, series_from: int = 0):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    return job.public(series_from)


@api.post("/jobs/{job_id}/stop", dependencies=[Depends(require_token)])
def stop(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    job.stop_flag = True
    return {"ok": True}


@api.get("/jobs/{job_id}/trajectory.pdb", response_class=PlainTextResponse, dependencies=[Depends(require_token)])
def trajectory(job_id: str, stride: int = 1):
    """Solute-only multi-model PDB — loads straight back into the Trajectory tab."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    if not job.frames:
        raise HTTPException(409, "no frames yet")

    template = [l for l in job.solute_pdb_header.splitlines()
                if l.startswith(("ATOM", "HETATM"))]
    out: list[str] = [f"REMARK   RNP Bench run {job.id}",
                      f"REMARK   {job.system.get('force_field','')}",
                      f"REMARK   frames are solute only, saved every "
                      f"{job.settings.get('save_ps')} ps"]
    for n, (coords, t_ps) in enumerate(
            zip(job.frames[::stride], job.frame_times[::stride]), start=1):
        out.append(f"MODEL     {n:4d}")
        out.append(f"REMARK   t = {t_ps} ps")
        for line, xyz in zip(template, coords):
            out.append(f"{line[:30]}{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{line[54:]}")
        out.append("ENDMDL")
    out.append("END")
    return "\n".join(out)


def write_dcd(job):
    """A binary trajectory that VMD, PyMOL, MDAnalysis and MDTraj all open."""
    if not job.frames or job.solute_topology is None or not job.work_dir:
        return None
    path = os.path.join(job.work_dir, "trajectory.dcd")
    dt_ps = float(job.settings.get("save_ps", 10.0))
    with open(path, "wb") as f:
        dcd = app.DCDFile(f, job.solute_topology, dt_ps * unit.picoseconds)
        for coords in job.frames:
            dcd.writeModel(coords * unit.angstrom)
    return path


def provenance(job) -> str:
    s = job.settings
    L = [
        "RNP Bench run provenance",
        "=" * 62,
        f"job id          {job.id}",
        f"name            {job.name}",
        f"started         {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(job.created))}",
        f"status          {job.status}",
        f"wall clock      {time.time() - job.created:.0f} s",
        f"throughput      {job.speed_ns_day:.2f} ns/day",
        "",
        f"OpenMM          {mm.version.version}",
        f"platform        {job.system.get('platform', '?')}  device {job.device}",
        f"hardware        {'; '.join(gpu_names()) or 'CPU'}",
        "",
        f"force field     {job.system.get('force_field', '?')}",
        f"water model     {s.get('water')}",
        f"total atoms     {job.system.get('total_atoms')}",
        f"solute atoms    {job.system.get('solute_atoms')}",
        f"waters          {job.system.get('waters')}",
        f"ions            {job.system.get('ions')}",
        f"net charge      {job.system.get('net_charge_e')} e",
        f"box (nm)        {job.system.get('box_nm')}",
        f"constraints     {job.system.get('constraints')}",
        f"H mass (amu)    {job.system.get('hydrogen_mass_amu')}",
        f"rebuilt atoms   {job.system.get('missing_atoms_added')} missing heavy atoms added",
        "",
        f"temperature     {s.get('temp_K')} K",
        f"pressure        {s.get('press_bar')} bar",
        f"time step       {s.get('dt_fs')} fs",
        f"heating         {s.get('heat_ps')} ps, solute restrained",
        f"equilibration   {s.get('equil_ps')} ps, restraints released in four stages",
        f"production      {s.get('prod_ps')} ps",
        f"frames saved    {len(job.frames)} every {s.get('save_ps')} ps",
        f"random seed     {s.get('seed') or 'not fixed'}",
        "",
        "What is in this archive",
        "-" * 62,
        "input.pdb             the structure exactly as submitted",
        "system_solvated.pdb   after hydrogens, water and ions were added",
        "system.xml            the fully parameterised OpenMM System",
        "integrator.xml        the integrator, including its seed",
        "final_state.xml       positions, velocities and box, for restarting",
        "final_frame.pdb       the last frame, every atom",
        "topology_solute.pdb   solute topology; the reference for the DCD",
        "trajectory.dcd        solute trajectory, binary",
        "trajectory.pdb        the same frames as a multi-model PDB",
        "series.csv            energy, temperature, density, volume against time",
        "log.txt               what the server did, with timings",
        "settings.json         every parameter this run used",
        "rerun.py              continues or repeats the run from system.xml",
        "",
        "Reproducing this run",
        "-" * 62,
        "system.xml holds the force field as it was actually applied, so you do not",
        "need the same force-field files installed to repeat it. Load it with",
        "final_state.xml to continue, or with system_solvated.pdb to start over:",
        "",
        "    python rerun.py --continue-from final_state.xml --ps 1000",
        "",
        "Opening the trajectory elsewhere",
        "-" * 62,
        "    import mdtraj",
        "    t = mdtraj.load('trajectory.dcd', top='topology_solute.pdb')",
        "",
        "    vmd topology_solute.pdb trajectory.dcd",
    ]
    if not s.get("seed"):
        L += ["", "NOTE: no random seed was fixed, so repeating this will not give the",
              "      same trajectory. Set a seed in the Run tab if you need that."]
    return "\n".join(L)


RERUN_SCRIPT = '''#!/usr/bin/env python3
"""Reproduce or continue this run.

Needs only OpenMM. The force field parameters travel inside system.xml, so no
force-field files have to be installed and no versions have to match.

    python rerun.py --continue-from final_state.xml --ps 1000
    python rerun.py --ps 1000            # start again from the solvated box
"""
import argparse
import openmm as mm, openmm.app as app
from openmm import unit

ap = argparse.ArgumentParser()
ap.add_argument("--continue-from", default=None)
ap.add_argument("--ps", type=float, default=1000.0)
ap.add_argument("--out", default="continued")
ap.add_argument("--save-ps", type=float, default=10.0)
args = ap.parse_args()

pdb = app.PDBFile("system_solvated.pdb")
system = mm.XmlSerializer.deserialize(open("system.xml").read())
integrator = mm.XmlSerializer.deserialize(open("integrator.xml").read())
sim = app.Simulation(pdb.topology, system, integrator)

if args.continue_from:
    sim.loadState(args.continue_from)
    print("continuing from", args.continue_from)
else:
    sim.context.setPositions(pdb.positions)
    print("minimising")
    sim.minimizeEnergy()
    sim.context.setVelocitiesToTemperature(300 * unit.kelvin)

dt = integrator.getStepSize()
steps = int(round(args.ps * unit.picoseconds / dt))
every = max(1, int(round(args.save_ps * unit.picoseconds / dt)))
sim.reporters.append(app.DCDReporter(args.out + ".dcd", every))
sim.reporters.append(app.StateDataReporter(
    args.out + ".csv", every, step=True, time=True, potentialEnergy=True,
    kineticEnergy=True, temperature=True, density=True, volume=True, speed=True))
print("running", steps, "steps")
sim.step(steps)
sim.saveState(args.out + "_state.xml")
print("done ->", args.out + ".dcd")
'''


# --------------------------------------------------------------------------
# Interaction energy along the trajectory (MM/GBSA, single trajectory)
#
# This is an interaction energy, NOT a binding free energy. There is no entropy
# term, one trajectory is used for all three states, and the GBn2 model here
# carries no salt screening -- which matters a great deal for a nucleic acid.
# Absolute values are not affinities. The change over time is the signal.
# --------------------------------------------------------------------------
GB_XML = "implicit/gbn2.xml"


def _subset(topology, positions, keep_atoms):
    m = app.Modeller(topology, positions)
    keep = set(keep_atoms)
    m.delete([a for a in m.topology.atoms() if a.index not in keep])
    return m.topology, m.positions


def _cross_terms(system, prot, rna):
    """van der Waals and Coulomb BETWEEN the two groups, evaluated directly.

    Taking E(complex) - E(protein) - E(RNA) means subtracting intramolecular
    energies of several hundred kcal/mol to reach an answer of a few, and in
    single precision the cancellation eats the result. An interaction group
    computes only the cross pairs, so nothing has to cancel.
    """
    nb = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    q, sig, eps = [], [], []
    for i in range(nb.getNumParticles()):
        qi, si, ei = nb.getParticleParameters(i)
        q.append(qi.value_in_unit(unit.elementary_charge))
        sig.append(si.value_in_unit(unit.nanometer))
        eps.append(ei.value_in_unit(unit.kilojoule_per_mole))

    vdw = mm.CustomNonbondedForce(
        "4*epsilon*((sigma/r)^12-(sigma/r)^6);"
        "sigma=0.5*(sigma1+sigma2); epsilon=sqrt(epsilon1*epsilon2)")
    vdw.addPerParticleParameter("sigma")
    vdw.addPerParticleParameter("epsilon")
    coul = mm.CustomNonbondedForce("138.935456*q1*q2/r")
    coul.addPerParticleParameter("q")
    for i in range(nb.getNumParticles()):
        vdw.addParticle([sig[i], eps[i]])
        coul.addParticle([q[i]])
    # OpenMM requires every force in a system to share the same exclusions.
    # They are all intramolecular, so they never touch the cross pairs anyway.
    for i in range(nb.getNumExceptions()):
        p1, p2, _q, _s, _e = nb.getExceptionParameters(i)
        vdw.addExclusion(p1, p2)
        coul.addExclusion(p1, p2)
    for f in (vdw, coul):
        f.setNonbondedMethod(mm.CustomNonbondedForce.NoCutoff)
        f.addInteractionGroup(prot, rna)   # cross pairs only
    vdw.setForceGroup(3); coul.setForceGroup(4)
    system.addForce(vdw); system.addForce(coul)
    return system


def _context(forcefield, topology, positions, cross=None):
    system = forcefield.createSystem(topology, nonbondedMethod=app.NoCutoff,
                                     constraints=None, rigidWater=False)
    for f in system.getForces():
        if isinstance(f, mm.NonbondedForce):
            f.setForceGroup(1)
        elif isinstance(f, mm.CustomGBForce) or "GB" in f.__class__.__name__:
            f.setForceGroup(2)
        else:
            f.setForceGroup(0)
    if cross is not None:
        _cross_terms(system, cross[0], cross[1])
    ctx = mm.Context(system, mm.VerletIntegrator(0.001),
                     mm.Platform.getPlatformByName("CPU"))
    ctx.setPositions(positions)
    return ctx


def run_energy(job, protein_chains, rna_chains, stride):
    KCAL = unit.kilocalorie_per_mole
    try:
        job.energy["status"] = "running"
        top = job.solute_topology
        if top is None or not job.frames:
            raise ValueError("this job has no stored frames")

        pc, rc = set(protein_chains), set(rna_chains)
        prot, rna = [], []
        for ch in top.chains():
            target = prot if ch.id in pc else (rna if ch.id in rc else None)
            if target is None:
                continue
            for r in ch.residues():
                for a in r.atoms():
                    target.append(a.index)
        if not prot or not rna:
            raise ValueError(f"selection is empty: {len(prot)} protein atoms, "
                             f"{len(rna)} nucleic atoms. Chains present: "
                             f"{[c.id for c in top.chains()]}")
        job.energy["note"] = f"{len(prot)} protein and {len(rna)} nucleic atoms"

        ff = app.ForceField(*FFDEF[job.settings["ff"]]["xml"][:-1], GB_XML)
        vec = lambda arr: [mm.Vec3(*x) for x in arr / 10.0] * unit.nanometer

        # complex: one context, with the cross terms as their own force groups
        cplx_atoms = sorted(prot + rna)
        remap = {a: k for k, a in enumerate(cplx_atoms)}
        t_c, p_c = _subset(top, vec(job.frames[0]), cplx_atoms)
        ctx_c = _context(ff, t_c, p_c,
                         cross=([remap[a] for a in prot], [remap[a] for a in rna]))
        # the two parts, needed only for the solvation difference
        t_p, p_p = _subset(top, vec(job.frames[0]), prot)
        t_r, p_r = _subset(top, vec(job.frames[0]), rna)
        ctx_p = _context(ff, t_p, p_p)
        ctx_r = _context(ff, t_r, p_r)
        idx_p, idx_r = sorted(prot), sorted(rna)
        job.energy["note"] += "; cross terms evaluated directly"

        frames = list(range(0, len(job.frames), max(1, stride)))
        out = {"t_ps": [], "vdw": [], "elec": [], "solv": [], "total": []}

        def gb(ctx, coords, idx):
            ctx.setPositions(vec(coords[idx]))
            return ctx.getState(getEnergy=True,
                                groups={2}).getPotentialEnergy().value_in_unit(KCAL)

        for n, fi in enumerate(frames):
            if job.stop_flag:
                job.energy["status"] = "stopped"
                return
            c = job.frames[fi]
            ctx_c.setPositions(vec(c[cplx_atoms]))
            st = lambda g: ctx_c.getState(getEnergy=True,
                        groups={g}).getPotentialEnergy().value_in_unit(KCAL)
            vdw, elec = st(3), st(4)
            solv = st(2) - gb(ctx_p, c, idx_p) - gb(ctx_r, c, idx_r)
            out["t_ps"].append(job.frame_times[fi])
            out["vdw"].append(round(vdw, 3))
            out["elec"].append(round(elec, 3))
            out["solv"].append(round(solv, 3))
            out["total"].append(round(vdw + elec + solv, 3))
            job.energy["progress"] = (n + 1) / len(frames)
            job.energy["series"] = out

        job.energy["series"] = out
        job.energy["status"] = "done"
        mean = sum(out["total"]) / len(out["total"])
        job.note(f"interaction energy over {len(frames)} frames: "
                 f"mean {mean:.1f} kcal/mol")
    except Exception as exc:
        job.energy["status"] = "failed"
        job.energy["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()


class EnergyRequest(BaseModel):
    protein_chains: list[str] = []
    rna_chains: list[str] = []
    stride: int = 1


@api.post("/jobs/{job_id}/energy", dependencies=[Depends(require_token)])
def start_energy(job_id: str, req: EnergyRequest):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    if job.energy.get("status") == "running":
        raise HTTPException(409, "an energy analysis is already running")
    job.energy = {"status": "queued", "progress": 0.0, "series": None,
                  "error": "", "note": "", "stride": req.stride}
    threading.Thread(target=run_energy,
                     args=(job, req.protein_chains, req.rna_chains, req.stride),
                     daemon=True).start()
    return {"ok": True}


@api.get("/jobs/{job_id}/energy", dependencies=[Depends(require_token)])
def get_energy(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    return job.energy


@api.get("/jobs/{job_id}/bundle.zip", dependencies=[Depends(require_token)])
def bundle(job_id: str, stride: int = 1):
    """Everything used and everything produced, in one archive."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    if not job.work_dir or not os.path.isdir(job.work_dir):
        raise HTTPException(409, "this job kept nothing on disk")

    if job.frames and not os.path.exists(os.path.join(job.work_dir, "trajectory.dcd")):
        try:
            write_dcd(job)
        except Exception as exc:
            print("DCD write failed:", exc)

    buf = io.BytesIO()
    root = f"{job.name}_{job.id[:8]}"
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for fname in sorted(os.listdir(job.work_dir)):
            path = os.path.join(job.work_dir, fname)
            if os.path.isfile(path) and fname != "frames.npy":
                z.write(path, f"{root}/{fname}")
        if job.frames:
            z.writestr(f"{root}/trajectory.pdb", trajectory(job_id, stride))
        z.writestr(f"{root}/series.csv", series_csv(job_id))
        z.writestr(f"{root}/log.txt",
                   "\n".join(f"{l['t']:8.1f}s  {l['text']}" for l in job.log))
        z.writestr(f"{root}/README.txt", provenance(job))
        z.writestr(f"{root}/rerun.py", RERUN_SCRIPT)
    data = buf.getvalue()
    return Response(data, media_type="application/zip", headers={
        "Content-Disposition": f'attachment; filename="{root}.zip"',
        "Content-Length": str(len(data))})


@api.get("/jobs/{job_id}/series.csv", response_class=PlainTextResponse, dependencies=[Depends(require_token)])
def series_csv(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    s = job.series
    rows = ["time_ps,stage,restraint_k,potential_kJ_mol,kinetic_kJ_mol,"
            "temperature_K,volume_nm3,density_g_cm3"]
    for i in range(len(s["t_ps"])):
        rows.append(f'{s["t_ps"][i]},{s["stage"][i]},{s["restraint_k"][i]},'
                    f'{s["pe"][i]},{s["ke"][i]},{s["temp"][i]},'
                    f'{s["volume"][i]},{s["density"][i]}')
    return "\n".join(rows)


@api.get("/")
def index():
    page = os.path.join(HERE, "rnp-bench.html")
    if os.path.exists(page):
        return FileResponse(page)
    return JSONResponse({"error": "rnp-bench.html not found next to rnp_server.py"}, 404)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1",
                    help="127.0.0.1 keeps it private (default). 0.0.0.0 exposes it "
                         "to the network — only do that behind TLS with a token.")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--token", default=None,
                    help="API token. Generated automatically when --host is not "
                         "localhost. Pass 'none' to disable (localhost only).")
    ap.add_argument("--allow-origin", action="append", default=[],
                    help="Browser origin permitted to call this server. Repeatable. "
                         "Required when serving the page from somewhere else.")
    ap.add_argument("--gpu", default="0",
                    help="GPU device indices, comma separated. One job runs per "
                         "device, so --gpu 0,1 allows two concurrent runs.")
    ap.add_argument("--max-atoms", type=int, default=400_000)
    ap.add_argument("--work-dir", default=os.path.join(HERE, "rnp_runs"))
    args = ap.parse_args()

    CFG.host = args.host
    CFG.origins = args.allow_origin
    CFG.devices = [d.strip() for d in args.gpu.split(",") if d.strip()] or ["0"]
    CFG.max_atoms = args.max_atoms
    CFG.work_dir = args.work_dir
    os.makedirs(CFG.work_dir, exist_ok=True)

    if args.token == "none":
        CFG.token = ""
    elif args.token:
        CFG.token = args.token
    elif not CFG.local_only():
        CFG.token = secrets.token_urlsafe(24)      # never exposed without one

    install_cors()

    plats = [mm.Platform.getPlatform(i).getName()
             for i in range(mm.Platform.getNumPlatforms())]
    gpus = gpu_names()

    print("=" * 66)
    print(f"  RNP Bench backend {VERSION}   OpenMM {mm.version.version}")
    print(f"  platforms : {', '.join(plats)}")
    print(f"  hardware  : {'; '.join(gpus) if gpus else 'CPU only — no GPU platform found'}")
    print(f"  slots     : {len(CFG.devices)} concurrent job(s) on device(s) {','.join(CFG.devices)}")
    print(f"  work dir  : {CFG.work_dir}")
    print("=" * 66)

    if CFG.local_only():
        print(f"\n  Listening on localhost only.")
        print(f"  Open  http://{args.host}:{args.port}/")
        print(f"\n  Running this on a remote GPU machine? Do not change --host.")
        print(f"  Forward the port instead — SSH is the authentication, and nothing")
        print(f"  is exposed to the network:")
        print(f"\n      ssh -N -L {args.port}:localhost:{args.port} USER@THIS_HOST\n")
        print(f"  then open http://localhost:{args.port}/ on your own machine.")
    else:
        print(f"\n  *** Listening on {args.host}:{args.port} — reachable from the network ***")
        if CFG.token:
            print(f"\n  API token:  {CFG.token}")
            print(f"  Paste it into the Token box in the Run tab. Requests without it")
            print(f"  are refused.")
        else:
            print(f"\n  *** NO TOKEN. Anyone who can reach this port can submit jobs,")
            print(f"      read trajectories and occupy the GPU. ***")
        if not CFG.origins:
            print(f"\n  No --allow-origin given, so browsers on other origins are blocked.")
            print(f"  Serving the page from this server at / still works.")
        print(f"\n  This speaks plain HTTP. Put it behind a reverse proxy with TLS")
        print(f"  before it crosses anything you do not control.")
    print()

    import uvicorn
    uvicorn.run(api, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
