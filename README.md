# RNP Bench

A workbench for RNA–protein molecular dynamics: prepare the system, look at the
physics, run the simulation, analyse the trajectory, and have Gemini review the
result and propose changes you approve one at a time.

Two files, and one optional helper:

| File | What it is |
|---|---|
| `rnp-bench.html` | the whole application — parsing, viewer, analysis, input-file generation, physics. Works on its own. |
| `rnp_server.py` | optional local backend that runs real all-atom MD with OpenMM. |
| `make_web_example.py` | optional: turns a finished run into an example that a public copy of the page can replay. |

Nothing leaves your machine except, if you choose to use it, a few kilobytes of
summary numbers sent to Gemini. Structures and trajectories never do.

---

## Updating an existing install

The page you see comes from **the machine running the server**, not from your
laptop. Copying only one of the two files, or not restarting, is the usual reason
a new feature does not appear.

```bash
scp rnp-bench.html rnp_server.py you@gpu-server:~/rnp/
ssh you@gpu-server
cd ~/rnp
# stop the old server with Ctrl-C, then
python rnp_server.py
```

Then **hard refresh** the browser: Ctrl-Shift-R, or Cmd-Shift-R on a Mac.

Both files carry a version stamp. The page shows its own next to the title, the
server prints its own at startup and reports it from `/health`, and step 7 refuses
to run and tells you plainly if the two do not match.

Sanity check from the remote machine:

```bash
curl -s localhost:8000/health | grep -o '"version":"[^"]*"'
curl -s localhost:8000/ | grep -c torsioncard      # 1 means the new page is being served
```

---

## Quick start

### Without the backend

Open `rnp-bench.html` in a browser. Everything works except running the
simulation and the Gemini review (see the note about `file://` below).

### With the backend — this is the recommended way

```bash
pip install "openmm>=8.1" pdbfixer fastapi "uvicorn[standard]" numpy
python rnp_server.py
```

Then open **http://localhost:8000/** — the server hands you the page, so
everything is same-origin and both the MD backend and Gemini work.

Keep both files in the same directory.

### GPU

`pip install openmm` gives you a CPU build. That is fine for a few thousand
atoms and far too slow for a solvated complex. For real throughput install a
CUDA build:

```bash
conda install -c conda-forge openmm cuda-version=12
```

The Run tab reports which platform it found. If it says CPU only, believe it.

---

## Running on a remote GPU machine

### The way you should do it: SSH tunnel

No configuration, nothing exposed, SSH is the authentication.

```bash
# on the GPU machine
python rnp_server.py                       # stays bound to 127.0.0.1

# on your laptop
ssh -N -L 8000:localhost:8000 you@gpu-server
```

Open `http://localhost:8000/`. The page is served from the remote machine, the
GPU does the work, and the port is never open to anything. For one person this
is the whole answer.

Add `--gpu 0,1` on a multi-GPU box to allow one concurrent job per device.

### If it genuinely has to listen on the network

```bash
python rnp_server.py --host 0.0.0.0 --gpu 0,1 --max-atoms 300000
```

The server refuses to be casual about this:

- **A token is generated automatically** whenever `--host` is not localhost, and
  printed in the startup banner. Every endpoint except `/health` requires it,
  as an `x-api-token` header or a `?token=` query parameter. Paste it into the
  **Token** box next to the backend URL in step 7.
- **CORS stops being a wildcard.** On a network interface only origins you name
  with `--allow-origin` are allowed. Serving the page from the server itself at
  `/` still works, because that is same-origin.
- **`--max-atoms`** rejects systems bigger than the machine should attempt,
  before anything is built.
- **Long series are decimated** and frames spill to `--work-dir`, so a 500 ns run
  does not sit entirely in RAM.

It still speaks plain HTTP. **Put it behind a reverse proxy with TLS** before it
crosses any network you do not control. A token over unencrypted HTTP is a token
anyone on the path can read. Minimal nginx:

```nginx
location / {
    proxy_pass http://127.0.0.1:8000;
    proxy_read_timeout 3600s;      # runs are long; do not let the proxy hang up
}
```

and start the backend on `127.0.0.1` so only the proxy can reach it.

### What it still is not

This is a single-tenant instrument, not a shared service. Jobs live in memory and
are lost if the process restarts, there is one token rather than user accounts,
and nothing stops a legitimate user from filling the disk. If several people need
it at once, put it behind a scheduler instead — see below.

---

## Publishing the page on a static website

`rnp-bench.html` can live on GitHub Pages or any other static host. Everything
except step 7 runs in the visitor's browser, so a public copy is a working tool,
not a screenshot. The backend does not move: it stays on the GPU machine, bound
to localhost, and the public page never connects to it.

The page works out at startup where a backend can be, and shows or hides **step
7, Run**, accordingly:

| Served from | Steps | What the page does |
|---|---|---|
| `rnp_server.py` itself, directly or through an SSH tunnel on any port | 8, with Run | finds the API at its own address and connects |
| `file://`, or a static server on `localhost` | 8, with Run | tries `http://localhost:8000`, as before |
| a public static host | 7, no Run | probes nothing on the visitor's machine; step 6 explains the handoff |

The last row matters twice over. A public page that probes `http://localhost`
makes current browsers ask the visitor for permission to reach their local
network, which is alarming and pointless. And a Run button that cannot run
anything is worse than no button: it promises something the page cannot deliver.

### What the seven-step build looks like

Nothing is deleted and there is no second file to keep in sync. The Run view
stays in the document; the parts of it that work without a backend move to where
they are still useful:

- **The Run tab disappears** and the remaining steps renumber to 1–7, so
  Trajectory becomes step 7.
- **The pre-flight force-field check moves to step 6**, next to the input files,
  under a short explanation of where those files go and what to bring back.
  Knowing that a residue has no template is more useful *before* you queue a job
  on a cluster than after.
- **Step 7, Trajectory, leads with a drop zone.** Bringing frames back is the
  first thing you do there, so it is the first thing on the page, with the
  `gmx trjconv` and `cpptraj` recipes for producing a multi-model PDB, and a
  reminder to strip the solvent before you write one. It collapses to a single
  line once frames are loaded.
- **The live plots move to step 7** and become a run log rather than a run in
  progress. They fill from a recorded example, or from a `series.csv` the visitor
  loads themselves — the same file the run archive contains.
- **Interaction energy stays visible but explains itself.** It needs the charges
  and Lennard-Jones parameters as applied, which a trajectory file does not
  carry, so it says so instead of failing.

Everything else — loading, structure checks, the interface, the physics, the
system plan, the input files, and the whole trajectory analysis — is unchanged,
because none of it ever needed a server.

To force the choice rather than detect it, edit one line near the top of the
script in `rnp-bench.html`:

```js
const RUN_MODE = 'auto';    // 'on' keeps the Run step, 'off' removes it
```

### An example run for visitors

Visitors cannot start simulations, so show them one.

1. Run something small on the GPU — the 1URN B + Q pair below is a good choice —
   and press **Download everything** in step 7.
2. Turn the archive into web files:

   ```bash
   python make_web_example.py rnp_1a2b3c4d.zip --title "1URN, chains B and Q"
   ```

   This writes `rnp-bench-example/` with `manifest.json`, `trajectory.pdb` and
   `series.csv`, using only the standard library. The trajectory is thinned to at
   most 200 frames and 20 MB (`--max-frames`, `--max-mb`), because github.com
   refuses browser uploads over 25 MB; `--no-hydrogens` roughly halves the file.
   Nothing else from the archive is published: no `system.xml`, no logs, no job id.
3. Upload the folder next to `rnp-bench.html`.

The page finds it by itself and shows **Load the example run** in step 1. The
frames open in the Trajectory step for the full analysis, and the recorded curves
appear there too, labelled as a recording. Interaction energy is the one panel
that stays unavailable, because it needs the backend that ran the system.

### Links that open something

- `rnp-bench.html?pdb=1URN` fetches that entry from RCSB on arrival.
- `rnp-bench.html?example` opens the example run.

### Using the public copy with your own backend

You rarely need to: open `http://localhost:8000/` through the tunnel and the
server hands you the same page, Run step included. If you do want the public copy, forward the port
as usual and press **Connect**; the browser may ask once whether the site may
reach your local network.

While a tunnel is open, a backend on localhost accepts requests from any origin,
so any page you have open could try to reach it — and once the code is public, so
is the API. When you connect a public copy, name its origin and set a token:

```bash
python rnp_server.py --allow-origin https://you.github.io --token <a long random string>
```

then paste the same string into the **Token** box. Never put a token in the page
or in the repository.

---

## A worked example: 1URN

`1URN` is the U1A spliceosomal protein bound to its 21-nucleotide RNA hairpin —
the textbook RRM–RNA complex, and small enough to actually run. Everything in it
is a standard residue, so nothing will hit a missing template.

**What is in the file:** three copies of the complex. Chains A, B, C are the
97-residue protein; chains P, Q, R are the 21-nt RNA (AAUCCAUUGCACUCCGGAUUU, all
standard A/U/G/C). Plus glycerol from the cryoprotectant and crystallographic
waters, both of which are stripped automatically.

**Run it like this:**

1. **Load** — type `1URN` into the fetch box, or download it from RCSB and drop
   the file in.
2. **Structure** — you will see six chains. Use the chain chips to hide
   **A, C, P and R**, leaving the **B + Q** pair. That is one complex instead of
   three, and it is the same monomer a published MD study used as its starting
   structure. Whatever you hide here is excluded from the run.
3. **Interface** — set protein chain B against RNA chain Q. You should see the
   aromatic stacking that defines RRM recognition: Tyr13 and Phe56 against the
   loop bases, plus hydrogen bonds to the AUUGCAC consensus.
4. **Physics** — the RNA survey should read A-form, C3′-endo, χ anti. That is
   your baseline; if it still reads that way at the end of the run, the force
   field behaved.
5. **System plan** — ff14SB/χOL3/TIP3P and 10 Å padding is a sensible start.
6. **Input files** — generated for all three engines, if you want to run it
   properly elsewhere.
7. **Run** — press **Build system only** first. It takes seconds and tells you
   the atom count, net charge and box before you commit. Expect roughly 25–30k
   atoms. Then start the run.
8. **Trajectory** — load the frames back and analyse.

**How long it takes.** About 28,000 atoms is roughly 1 ns/day on CPU and a few
hundred ns/day on a decent GPU. For a first end-to-end pass on CPU, set
equilibration to 20 ps and production to 50 ps — that finishes in minutes and
exercises every step. It is far too short to mean anything scientifically, and
the report sent to Gemini will say so.

**If you want something smaller still,** drop the padding to 8 Å and keep only
chains B and Q; that is the smallest honest version of this system.

---

## Will it build? Ask before you run

Step 7 checks every residue against the force field before anything starts, and
tells you which of three situations you are in:

- **stripped automatically** — waters, glycerol, sulfate and other crystallisation
  junk;
- **renamed** — RNA written as `RA`/`RC5`/`ADE` in AMBER or CHARMM conventions is
  mapped onto the `A`/`C`/`G`/`U` templates for you;
- **a blocker** — modified nucleotides such as pseudouridine, ligands you want to
  keep, or metals beyond Na/K/Cl/Mg. These are named individually, with what to
  do about each.

This is why tRNA structures fail: they are full of modified bases (Ψ, m¹A, m⁵U)
that have no template in the standard set. The check names them instead of
letting OpenMM throw twenty seconds into the build.

---

## The eight steps

1. **Load** — PDB, mmCIF or GROMACS `.gro`. Multi-model files load as a trajectory.
2. **Structure** — cartoon viewer with secondary structure from backbone hydrogen
   bonds, composition, sequences, and pre-flight checks (gaps, missing side
   chains, non-standard residues, clashes).
3. **Interface** — contacts, salt bridges to phosphate, hydrogen bonds, stacking,
   Watson–Crick pairs, buried surface area from a Shrake–Rupley SASA.
4. **Physics** — collective motions, electrostatics, RNA sugar and backbone
   geometry, Ramachandran. Details below.
5. **System plan** — force field, box, water, ions, cost estimate, protocol.
6. **Input files** — GROMACS, AMBER and OpenMM inputs, downloadable as a zip.
7. **Run** — real MD on the backend, with temperature, energy, density and box
   volume plotted live. Frames load straight back into step 8, and
   **Download everything** gives you the whole run as one archive.
8. **Trajectory** — the Motion panel plays the trajectory in 3D, then RMSD,
   RMSF, radius of gyration, contact and hydrogen-bond occupancy, base-pair
   survival, and the Gemini review panel.

---

## RNA torsions over time

In step 8. Works on any trajectory, including one loaded from a file — it is pure
geometry and needs no force field.

- **A-form retention** — the percentage of nucleotides in C3′-endo, anti χ and
  high-anti χ at every frame. This is the single most useful plot for an RNA run:
  if C3′-endo drains away while high-anti χ fills up, you are watching the ff99
  ladder artefact, which is a force-field problem rather than a sampling one.
- **One angle for one residue** — α, β, γ, δ, ε, ζ, χ or the pseudorotation phase
  P, against time. **Unwrap** is on by default so a wrap through 360° does not
  draw a spurious vertical line.
- **Spread per residue** — circular standard deviation. Angles live on a circle,
  so a plain mean of 359° and 1° would give 180°; everything here uses circular
  statistics.
- **Flip counts** — how many times each nucleotide crossed between North and
  South pucker, or between syn and anti. A residue can have a large spread
  because it is noisy or because it is switching between two states; the flip
  count is what distinguishes them. Rows that switch often are highlighted.

CSV export gives every angle for every residue at every frame.

---

## Interaction energy over time

Also step 8, but this one **needs a completed backend run** — it uses the charges
and Lennard-Jones parameters that were actually applied, which a trajectory
loaded from a file does not carry.

Reported per frame: van der Waals, electrostatic, GBn2 solvation, and the total.

**It is an interaction energy, not a binding free energy.** There is no entropy
term, all three states come from one trajectory, and the GB model used here has
no salt screening — which matters a great deal for a phosphate backbone. Do not
quote the absolute number as an affinity. What is worth reading is how it moves.

Two implementation notes, because they change the numbers:

- The protein–RNA van der Waals and Coulomb terms are computed **directly**, with
  an OpenMM interaction group, rather than as E(complex) − E(protein) − E(RNA).
  Subtracting intramolecular energies of several hundred kcal/mol to reach an
  answer of a few loses the answer to rounding. Measured on a separated pair, the
  direct route leaves a noise floor of about 0.02 kcal/mol; the subtraction route
  was off by orders of magnitude.
- The drift test compares the two halves of the run against the scatter **within**
  each half, not the scatter of the whole series. A genuine step change inflates
  the whole-series spread and would otherwise hide itself.

Consecutive MD frames are correlated, so any significance the panel suggests is
optimistic. Treat it as a flag to go and look at the contact occupancy, not as a
test result.

---

## Download everything

**Download everything (.zip)** in step 7 is a reproducibility archive, not just
coordinates:

| File | Why it is there |
|---|---|
| `input.pdb` | the structure exactly as submitted |
| `system_solvated.pdb` | after hydrogens, water and ions |
| `system.xml` | the **fully parameterised** OpenMM System — every charge and bonded term as actually applied |
| `integrator.xml` | the integrator, including its seed |
| `final_state.xml` | positions, velocities and box, for restarting |
| `trajectory.dcd` + `topology_solute.pdb` | binary trajectory for VMD, PyMOL, MDAnalysis, MDTraj |
| `trajectory.pdb` | the same frames as a multi-model PDB |
| `series.csv` | energy, temperature, density, volume against time |
| `log.txt`, `settings.json`, `README.txt` | what happened, with what settings, on what hardware |
| `rerun.py` | continues or repeats the run |

`system.xml` is the point. It holds the force field **as applied**, so repeating
the run needs neither the same force-field files nor the same OpenMM version:

```bash
python rerun.py --continue-from final_state.xml --ps 1000
```

```python
import mdtraj
t = mdtraj.load('trajectory.dcd', top='topology_solute.pdb')
```

Set a **seed** in the Run tab if you want a bit-for-bit repeat. Left at 0 every
run differs, and `README.txt` says so.

---

## The Motion panel

The same viewer moves into step 8 rather than a second one being created, so the
representation, colouring, chain selection and camera you set in step 2 carry
over.

- **Superpose on frame 1** — on by default, and the thing that matters most.
  Without it, almost everything you see is the molecule tumbling in the box
  rather than changing shape. Turn it off and the panel says so.
- **Ghost of frame 1** — the starting structure drawn faintly behind the current
  one, so you compare rather than remember.
- **Colour by shift** — per-residue heavy-atom displacement from the reference,
  updated live. This is where a loop opening or an interface loosening shows up.
- **Smooth** — interpolates between saved frames, so 10 ps spacing still plays
  fluidly.
- **Detail** — spline density for the cartoon. *Maximum* roughly triples the
  ribbon resolution; drop to *standard* if playback stutters on a big system.
- Loop, rock and once, with speed from ¼× to 4×. Rendering drops to a fast path
  while playing and returns to full quality the moment you pause.

---

## What the Physics tab actually computes

**Collective motions.** Every residue is a node, every close pair a spring.
Diagonalising the network gives the slow collective motions with no sampling at
all. The honest test is the crystallographic B-factors, which the model never
sees: the tab reports the Pearson correlation and the cutoff you used, because
the fit genuinely depends on the cutoff. Values around 0.4–0.7 are normal;
crystal packing restrains surface loops that an isolated-molecule network leaves
free. If the network splits into disconnected pieces the tab says so rather than
quietly analysing a fragment.

**Electrostatic potential.** Screened Coulomb potential from formal charges —
phosphates, carboxylates, Lys, Arg, resolved ions — with Debye–Hückel screening
at the ionic strength you set. Colour the structure by it and you see why raising
the salt weakens a phosphate-driven interface. This shows where the field is; it
is not a binding free energy.

**RNA sugar and backbone.** Pseudorotation phase and amplitude by Altona and
Sundaralingam, glycosidic χ, and the α–ζ backbone torsions. A large high-anti χ
population together with a depleted C3′-endo population is the signature of the
ff99 ladder artefact — the thing χOL3 exists to prevent. The tab flags it.

**Live energy and temperature.** During a run you watch the potential energy
fall, the kinetic energy rise, and the temperature settle. It will dip well below
your target in the first picoseconds: that is equipartition draining kinetic
energy into potential as the system relaxes, and the thermostat pulling it back.

---

## Gemini review

The panel is in step 8.

1. Get an API key from Google AI Studio.
2. Paste it into the panel. It is held in memory and never written anywhere.
3. Press **List models** to populate the dropdown from your key — model names
   change often, so this is more reliable than a hard-coded list.
4. Run the analysis, then press **Review this run**.

What is sent is a few kilobytes of summary numbers — you can see exactly what,
under "What gets sent". No coordinates, no frames.

The reply is constrained by a response schema to a fixed set of parameters, each
clamped to a sane range on the way back. **Nothing is applied automatically.**
Every proposal appears as a card with Apply and Reject, tagged as a setup fix,
more sampling, or a change to the science. Approvals and rejections both go to a
downloadable decision log.

The report also carries a `caveats` list generated locally — too few frames for a
drift statistic, no equilibration discarded, ion atmosphere not converged — so
the model is told what the numbers cannot support instead of inferring a trend
that is not there.

### Using it responsibly

A model that keeps adjusting settings until the answer looks good is a machine
for manufacturing false positives. Three habits prevent that:

- **Write down the success criterion before you start** and do not let it move.
- **Treat setup fixes and science changes differently.** "The box is too small,
  the run is too short" is the model doing its job. "Try a different temperature"
  is a decision you own.
- **Validate on fresh replicas.** Whatever settings you converge on, run them
  again from new velocities. Numbers from the runs you selected on do not count.

---

## Notes and limits

- **`file://` will bite you.** Opened from a file path, the browser sends
  `Origin: null` and blocks both the backend and the Gemini API. Serve the page
  over HTTP — the backend at `/` does this for you.
- **OpenMM ships ff14SB, not ff19SB.** If you pick ff19SB in step 5 the generated
  AmberTools and GROMACS inputs use it correctly, but a run on the backend
  substitutes ff14SB and says so in the log.
- **`missingResidues` is deliberately left empty.** Absent loops are not modelled
  in, because a guessed loop in an interface is worse than a visible gap. Model
  them yourself first if you need them.
- **Crystallographic waters are removed by default.** Turn that off if buried
  interface waters matter to you — for RNA–protein recognition they often do.
- **One job at a time.** The backend refuses a second run while one is going.

## Running the tests

```bash
node test_core.js        # parsing, superposition, SASA, geometry
node test_ss.js          # secondary structure against ideal helix and sheet
node test_physics.js     # pseudorotation, electrostatics, elastic networks
node test_e2e.js         # the whole page, headless        (needs: npm i jsdom)
python test_server.py    # the MD backend, end to end
node check_globals.js    # no duplicate names across the bundled files
```
