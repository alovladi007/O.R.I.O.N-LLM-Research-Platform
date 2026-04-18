# ORION Platform — Multi-Week Implementation Prompt Pack

**Scope:** Turn a prototype with production-grade scaffolding into a scientifically correct, end-to-end functional materials-discovery platform.
**Cadence:** 14 weekly phases, each with 3–5 atomic prompt sessions (~2–4 hrs of focused work each). Feed one prompt per session to Claude.
**Conventions:**
- Each prompt is self-contained — paste it verbatim to start a session.
- Each ends with an **"Acceptance tests"** block. Do not mark the session done until those pass.
- Canonical backend paths: `src/api/` (FastAPI app, routers) + `backend/common/` (domain logic). Frontend: `frontend/src/`.
- Before every session, run: `git checkout -b phase-N-session-M` and open a draft PR.

---

## PHASE 0 — Repository Hygiene & Ground Truth (Week 1)

Before any science: the codebase has 4 parallel API entry points, dead `src/` modules, 30+ overlapping status docs, and claims that don't match code. Sort this out first or every later session fights the mess.

### Session 0.1 — Entry-point consolidation

```
You are cleaning up the ORION repo. There are multiple FastAPI entry points:
  - src/api/app.py        (canonical, production)
  - src/api/app_dev.py    (dev variant)
  - simple_api.py         (top-level demo)
  - standalone_api.py     (if present)
  - demo_app.py           (top-level)
  - run_demo.py           (top-level)

Task:
1. Read each of these files and diff what they actually do (routes registered, middleware, startup hooks).
2. Confirm src/api/app.py is the canonical app — it should include every router under src/api/routers/.
3. For each non-canonical entry point, either:
   (a) delete it if redundant, or
   (b) move it to examples/ or scripts/ with a top-of-file comment explaining its purpose.
4. Update Makefile, README.md, BACKEND_QUICKSTART.md, and pyproject.toml console_scripts so they all point to `src.api.app:app` via uvicorn.
5. Verify frontend API_BASE_URL references resolve to the canonical backend port (8002 per recent commits).

Do NOT delete anything irreversibly — use `git rm` so it's in history. Leave a short note in SUMMARY.md listing what moved/was removed and why.

Acceptance tests:
- `uvicorn src.api.app:app --port 8002` starts without import errors.
- `curl http://localhost:8002/healthz` returns 200.
- `rg -n "simple_api|standalone_api|demo_app" --type py` returns zero hits except inside examples/.
- `pytest tests/ -x` still passes (should be trivially green — test suite is currently small).
```

### Session 0.2 — Kill the legacy `src/` modules or integrate them

```
The top-level src/ directory contains modules that predate the backend/common/ refactor:
  src/candidate_generation/, src/knowledge_graph/, src/rag/, src/data_ingest/,
  src/simulation/, src/experimental_design/, src/evaluation/, src/feedback_loop/,
  src/core/, src/ui/, src/worker/

Note: src/api/ is the CURRENT backend — do not touch it. Only audit the non-api siblings.

Task:
1. For each non-api src/ subpackage, run `rg -n "from src\.<pkg>|import src\.<pkg>"` across the whole repo.
2. Build a table: package | imported by (paths) | has tests | decision.
3. Decision rules:
   - Zero imports + no tests → delete the package.
   - Imported only by src/ui/streamlit_app.py → move under legacy/streamlit/ and mark deprecated in README.
   - Imported by backend/common or src/api → promote the module into backend/common/<domain>/ and update imports.
4. For the knowledge_graph module specifically: Neo4j is in docker-compose but unused. Either (a) wire it as an optional dependency behind a `ENABLE_KG=false` feature flag in src/api/config.py, or (b) remove Neo4j from docker-compose. Pick (b) unless you find real KG usage.
5. Run the full test suite and the uvicorn import smoke test after each batch of moves.

Acceptance tests:
- `python -c "import src.api.app"` works.
- docker-compose.yml has no orphan services.
- MIGRATION_GUIDE.md is updated with a "what moved" table at the top.
```

### Session 0.3 — Docs triage

```
There are ~30 markdown status files in the repo root, most contradicting each other
(SESSION_5_IMPLEMENTATION.md through SESSIONS_21-28_IMPLEMENTATION.md, plus
COMPLETE_IMPLEMENTATION_SUMMARY.md, IMPLEMENTATION_STATUS.md, etc.).

Task:
1. Create docs/history/ and move every SESSION_*.md, SESSIONS_*.md, CI-CD_FIX_*.md,
   GITHUB_PUSH_INSTRUCTIONS.md, PUSH_TO_GITHUB.md, CLEANUP_BRANCHES.md there.
2. Keep at the root only: README.md, DEPLOYMENT.md, QUICK_START.md, CHANGELOG.md (create if missing).
3. Rewrite README.md to reflect REALITY (not aspiration). Include sections:
   - What works end-to-end today (be honest — the structure→simulate→result path is broken)
   - Architecture diagram (ASCII or mermaid) of api + workers + db + engines
   - Local dev quickstart (the canonical path from Session 0.1)
   - Roadmap link to this ROADMAP_PROMPTS.md file
4. Create CHANGELOG.md seeded with a "0.1.0-prototype" entry summarizing current state.

Acceptance tests:
- `ls *.md` at repo root returns ≤6 files.
- README.md's quickstart, when followed literally, brings up a working API on port 8002.
- No section in README.md claims a feature that's stubbed (verify by grepping for the features in the routers).
```

### Session 0.4 — Secrets, CORS, and .env hygiene

```
Current security state:
  - .env is checked into git (verify with `git ls-files | grep -E '^\.env$'`).
  - .env.example has demo passwords like orion_secure_pwd / orion_redis_pwd.
  - src/api/config.py has cors_allow_methods: List[str] = ["*"] and likely cors_allow_origins = ["*"].
  - Multiple API keys (OpenAI, Anthropic, SMTP) are plain env vars.

Task:
1. If .env is tracked, run `git rm --cached .env` and ensure .gitignore covers it. Warn me BEFORE force-pushing history rewrites; for now just stop tracking forward.
2. Rotate any real-looking credentials — if you find anything that looks like a live secret (20+ char entropy in .env), flag it in the PR description and ask me to rotate.
3. In src/api/config.py:
   - cors_allow_origins: read from env CORS_ORIGINS as comma-separated list, default to ["http://localhost:3000","http://localhost:3002"].
   - cors_allow_methods: ["GET","POST","PUT","PATCH","DELETE","OPTIONS"].
   - cors_allow_credentials: True only if origins is not "*".
   - Reject ["*"] at startup with a clear error unless ORION_ENV=dev.
4. Introduce src/api/config.py secret validation: on startup, assert that JWT_SECRET, DATABASE_URL, REDIS_URL are set and JWT_SECRET is ≥32 chars. Fail loud if not.
5. Create docs/SECURITY.md describing the secret model, rotation policy, and that production must use a real secret manager (AWS Secrets Manager / Vault). Leave a TODO for Phase 13 to wire one.

Acceptance tests:
- `git ls-files .env` returns nothing.
- Starting the app with CORS_ORIGINS="*" and ORION_ENV=prod fails fast with a descriptive error.
- `curl -H "Origin: http://evil.com" -X OPTIONS http://localhost:8002/api/v1/materials` does NOT return Access-Control-Allow-Origin.
```

### Session 0.5 — Baseline test harness

```
Currently tests/ has only test_api.py and test_structures.py. Coverage is ~effectively zero on the paths that matter. Before we build more, establish a real test baseline so every later phase has a regression net.

Task:
1. Configure pytest:
   - Ensure pytest.ini / pyproject.toml has: asyncio_mode=auto, testpaths=["tests"], addopts="--strict-markers --tb=short -ra".
   - Add markers: unit, integration, slow, requires_db, requires_redis, requires_engines.
2. Add pytest-cov to requirements-dev.txt (create this file). Target: report only, no threshold yet.
3. Create tests/conftest.py with fixtures:
   - `anyio_backend` → "asyncio"
   - `db_session` → async SQLAlchemy session over a per-test transaction (rolled back on teardown). Use a disposable Postgres via testcontainers-python OR fall back to SQLite-in-memory with a skip marker for features that require Postgres-only (pgvector, JSONB GIN).
   - `api_client` → httpx.AsyncClient bound to the FastAPI app with auth bypass fixture.
   - `authenticated_user` → creates a user and yields a JWT.
4. Write smoke tests in tests/test_smoke.py covering:
   - GET /healthz, /readyz
   - POST /api/v1/auth/register, /auth/login (round-trip a token)
   - GET /api/v1/materials with auth
5. Add a GitHub Actions job that runs `pytest -m "not slow and not requires_engines"` on every PR.

Acceptance tests:
- `pytest -q` prints ≥6 passing tests.
- `pytest --cov=src --cov=backend --cov-report=term-missing` runs and prints a coverage number.
- CI job "tests" passes on a fresh PR.
```

---

## PHASE 1 — Data & Domain Foundation (Week 2)

The materials/structures/jobs models exist but APIs return mocks. We need a single honest data path before wiring any science on top.

### Session 1.1 — Structure parsing: wire `backend/common/structures/` to the API

```
Current bug: src/api/routers/structures.py has a `parse_structure_file` handler that returns hardcoded mock data. Meanwhile backend/common/structures/ contains real pymatgen-based parsers. Connect them.

Task:
1. Read backend/common/structures/__init__.py and identify the parser functions. Expected capability: parse CIF, POSCAR, XYZ, CSSR, PDB.
2. If the module does not already expose a unified entry point, add:
   ```python
   def parse_structure(content: str | bytes, fmt: str | None = None) -> pymatgen.Structure
   def structure_to_dict(s: pymatgen.Structure) -> dict  # ORION schema
   def export_structure(s: pymatgen.Structure, fmt: str) -> str
   ```
   `fmt` auto-detection uses file extension + content sniffing (CIF starts with `data_`, POSCAR has 2nd line as scale factor, etc.).
3. Rewrite src/api/routers/structures.py:
   - POST /api/v1/structures/parse accepts multipart file upload OR raw text body. Returns ORION structure schema (lattice, species, coords, spacegroup via pymatgen.symmetry).
   - POST /api/v1/structures/export takes a structure_id and target fmt; returns file bytes with correct content-type.
   - Both endpoints use the functions from backend/common/structures/.
4. Persist parsed structures into the DB via the existing Structure model. Compute and store:
   - formula_pretty, formula_anonymous
   - spacegroup_number, spacegroup_symbol, crystal_system
   - lattice params (a,b,c,α,β,γ), volume, density
   - n_atoms, n_species, reduced_formula
5. Scientific correctness notes to enforce in code (add docstrings citing the convention):
   - Fractional vs cartesian coords — store as fractional, return both via getters.
   - Lattice vectors in Å, angles in degrees.
   - Spacegroup via `SpacegroupAnalyzer(s, symprec=0.01, angle_tolerance=5)` — expose symprec as a request param with default 0.01.
   - Reject structures with overlapping sites (min pair distance < 0.5 Å) with a 422.

Acceptance tests:
- Write tests/test_structures_parse.py with real CIF fixtures for: NaCl (Fm-3m), Si (Fd-3m), graphene (P6/mmm via 2D unit cell), and a POSCAR for FCC Cu.
- Assert spacegroup numbers match: 225, 227, 191, 225 respectively (within symprec=0.01).
- Round-trip test: parse → export CIF → parse again → structures equivalent (use pymatgen.StructureMatcher).
- An invalid CIF returns 422 with a useful error message.
```

### Session 1.2 — Materials + Structures relational model

```
The Material and Structure models exist but their relationship and required columns need an audit.

Task:
1. Read src/api/models/ for Material, Structure, Property (and whatever else is there). List each column and its science meaning.
2. Ensure the schema supports:
   - A Material has many Structures (polymorphs, supercells, defects, surfaces).
   - Each Structure has a unique hash (structure_hash) computed from reduced composition + symmetrized sites to enable dedup. Use a stable canonicalization (pymatgen StructureMatcher's fingerprint or a custom SHA256 over sorted symmetrized coords).
   - Properties are associated with (Structure, Method, Conditions) triples — not with Material alone. A property MUST record:
     * method (DFT-PBE, DFT-HSE06, MD-LJ, experimental, etc.) as an enum + free-form functional/forcefield string
     * conditions (temperature_K, pressure_GPa, strain tensor) as JSONB
     * value + unit + uncertainty (+/- or distribution JSON)
     * provenance_job_id FK (nullable for imports)
3. Add Alembic migration for any missing columns. Use pgvector for an embedding column on Structure (dim=256, will be populated in Phase 6). Add GIN indexes on Material.tags, Property.conditions.
4. Backfill structure_hash for any existing rows.

Scientific correctness:
- Unit system: SI internally (K, Pa, J, m). Accept Å/eV/GPa at the API boundary with explicit conversion. Add a `backend/common/units.py` helper with pint registry.
- Composition normalization: always store reduced formula (gcd of stoichiometric coefficients).

Acceptance tests:
- Alembic upgrade+downgrade cycle is clean.
- tests/test_materials_model.py: creating two identical structures (down to symmetry) produces the same hash and a uniqueness violation on second insert.
- tests/test_units.py: 1.0 eV round-trips through the pint registry to 1.602176634e-19 J.
```

### Session 1.3 — Property ingestion API + CSV bulk importer

```
Scientists have CSV tables of measured/DFT-computed properties. Build a real ingestion path.

Task:
1. Add POST /api/v1/properties/bulk that accepts a CSV (or Parquet) and a schema mapping:
   ```json
   {
     "structure_ref": {"column": "mp_id", "kind": "external_id", "namespace": "materials_project"},
     "property": "formation_energy_per_atom",
     "value_column": "e_form",
     "unit": "eV/atom",
     "method": {"kind": "DFT", "functional": "PBE"},
     "conditions": {"temperature_K": 0, "pressure_GPa": 0}
   }
   ```
2. Support resolving structure_ref either by (a) existing ORION structure UUID, (b) Materials Project ID (via their free REST API if MP_API_KEY is set — behind a feature flag), or (c) uploaded CIF.
3. Background job — push the import to Celery with progress tracking (write to a BulkImportJob table with rows_total / rows_ok / rows_failed / errors JSONB).
4. Validation rules:
   - Reject rows where unit doesn't match the declared property's canonical unit (maintain a registry in backend/common/properties/registry.py: e.g., formation_energy → eV/atom, bandgap → eV, bulk_modulus → GPa).
   - Reject rows with value outside physical bounds (bandgap ≥ 0, density > 0, Poisson ratio in [-1, 0.5]). The bounds table goes in the registry.
5. Return a job ID; GET /api/v1/jobs/bulk-import/{id} reports progress and downloads the error CSV.

Acceptance tests:
- tests/fixtures/ contains a 100-row CSV of MP formation energies. Import → all rows land in DB within 10s (local).
- A CSV with bandgap=-1.2 eV for one row is rejected at that row only; the rest import.
- Downloading errors.csv returns exactly the rejected rows with an `error` column.
```

### Session 1.4 — Jobs model becomes the source of truth

```
src/api/routers/jobs.py has stubs; src/api/models/ has a Job model. Jobs are the spine of simulation orchestration — every engine run, ML training run, and agent campaign step is a Job. Make this real.

Task:
1. Job state machine: pending → queued → running → (succeeded | failed | cancelled | timeout). Enforce transitions in the model layer with a `transition_to(state)` method that raises on illegal transitions.
2. Job rows must carry:
   - kind (enum: dft_relax, dft_static, md_nvt, md_npt, ml_train, ml_infer, bo_suggest, al_query, import, export, agent_step)
   - inputs_json (validated against per-kind Pydantic schemas in backend/common/jobs/schemas.py)
   - outputs_json, artifacts_uri (MinIO path)
   - celery_task_id, worker_hostname, started_at, finished_at
   - cost_core_hours (set by worker)
   - parent_job_id (for DAG workflows)
3. Wire real cancellation: POST /api/v1/jobs/{id}/cancel should celery.control.revoke(task_id, terminate=False) for queued jobs and terminate=True for running ones. Update state to cancelled.
4. Add GET /api/v1/jobs/{id}/logs that streams the tail of the worker log file from MinIO (upload happens in Session 2.x).
5. Add an SSE endpoint GET /api/v1/jobs/{id}/events for live state transitions (this will power the frontend job monitor).

Acceptance tests:
- Illegal state transition (succeeded → running) raises and is rejected by the router with 409.
- A test that enqueues a fake 30s sleep task, cancels via API, and sees state=cancelled within 2s.
- SSE endpoint emits events when a second process updates the job row.
```

### Session 1.5 — Seed data: load a real Materials Project subset

```
We need enough real data to validate every subsequent phase. Load a curated subset from Materials Project.

Task:
1. Create scripts/seed_mp_subset.py that downloads via mp-api (pymatgen client) a specific curated subset:
   - ~200 oxides and perovskites (ABO3) spanning band gaps 0–6 eV
   - ~50 elemental metals (for benchmarking DFT)
   - ~50 2D materials (from MP 2D dataset)
   - For each: store structure (CIF), formation_energy_per_atom, bandgap, density, elastic_tensor if available.
2. Respect MP rate limits; cache downloads under data/mp_cache/ keyed by mp-id.
3. Idempotent: running twice doesn't duplicate.
4. Seed 3 users: admin@orion.dev, scientist@orion.dev, viewer@orion.dev with roles and hashed passwords from env.
5. Guard behind MP_API_KEY env var; skip gracefully with a warning if absent and load a bundled 20-structure offline fallback instead.

Acceptance tests:
- After running, `SELECT COUNT(*) FROM structures` ≥ 20 (offline) or ≥ 300 (with API key).
- `SELECT AVG(bandgap) FROM properties WHERE property='bandgap'` returns a physical number.
- Re-running the script produces 0 inserts and 0 errors.
```

---

## PHASE 2 — Job Execution Spine (Week 3)

Celery infra exists; actual submission-to-execution is stubbed. This phase makes a `POST /jobs` with `kind=mock` actually run on a worker, write artifacts, and surface results.

### Session 2.1 — Celery app, queues, and the base Task class

```
Task:
1. Consolidate Celery config into src/api/celery_app.py (or backend/common/workers/app.py — pick one, document choice). Queues:
   - default (catch-all)
   - dft (long CPU, 1 concurrency per worker)
   - md  (long CPU/GPU)
   - ml  (GPU if available)
   - io  (imports, exports, lightweight)
2. Base task class backend/common/workers/base.py:
   - on_start: mark job running, set worker_hostname, started_at
   - on_success: mark succeeded, persist outputs, upload artifacts
   - on_failure: mark failed, persist traceback, upload partial artifacts
   - on_retry: increment retries counter
   - emit Redis pubsub events for SSE (Session 1.4)
3. Artifact handling: every task gets a run_dir = tempfile.mkdtemp(prefix="orion-run-"). On completion, tar.gz the dir and upload to minio://artifacts/jobs/{job_id}/run.tgz. Store a manifest.json in the bundle.
4. Retry policy: DFT/MD use autoretry_for=(TransientEngineError,), max_retries=2, with exponential backoff (60s, 300s). ML retries once on CUDA OOM after scaling batch_size by 0.5.

Acceptance tests:
- A dummy task that sleeps 1s and writes a file in run_dir produces an artifact in MinIO.
- Killing the worker mid-task leaves the job in `running` for <30s then a beat-scheduled reaper flips it to failed with reason="worker_lost".
- SSE events for pending→running→succeeded observed in an integration test.
```

### Session 2.2 — Mock engine, end-to-end

```
Purpose: prove the spine works end-to-end before touching real DFT/MD.

Task:
1. Expose backend/common/engines/mock.py via the job dispatcher. It should accept a structure and return:
   - A fake energy = -sum(Z_i) * 1.5 eV/atom + 0.01 * noise
   - A fake force array = zeros + noise ~ 0.05 eV/Å
   - A fake trajectory of 10 frames (small displacements)
2. POST /api/v1/jobs with body {"kind":"mock_static","structure_id":"..."} creates a Job, enqueues the task, returns job_id.
3. Worker picks up → loads structure → runs mock engine → writes energy/forces to outputs_json → uploads trajectory.xyz to MinIO.
4. GET /api/v1/jobs/{id} returns outputs after completion.
5. Add /api/v1/jobs/{id}/artifacts that returns presigned MinIO URLs.

Acceptance tests:
- Integration test: submit 20 mock jobs in parallel, all finish within 60s, all have artifacts.
- Outputs schema validated against backend/common/jobs/schemas.py:MockStaticOutput.
- Frontend (even if minimal) can poll and display at least "status: succeeded, energy: X eV".
```

### Session 2.3 — Execution backends: local + SLURM

```
Real engines run either locally (small jobs) or on HPC via SLURM. Build a clean abstraction.

Task:
1. backend/common/execution/ backend interface:
   ```python
   class ExecutionBackend(Protocol):
       async def submit(self, cmd: list[str], run_dir: Path, resources: Resources) -> SubmissionHandle
       async def poll(self, handle) -> JobState
       async def cancel(self, handle) -> None
       async def fetch_artifacts(self, handle, run_dir) -> None
   ```
2. Implement LocalBackend (subprocess + psutil for memory/cputime) and SlurmBackend (sbatch/squeue/scancel with SSH via asyncssh if remote).
3. Resources dataclass: cpus, gpus, memory_gb, walltime_minutes, queue, account.
4. Wire into engine runners: QE/LAMMPS tasks no longer call subprocess directly — they call the execution backend chosen per-job from inputs.execution.kind (default: local).
5. Config: ORION_SLURM_HOST, ORION_SLURM_USER, ORION_SLURM_KEY_PATH, ORION_SLURM_PARTITION.

Acceptance tests:
- Unit test with a fake subprocess for LocalBackend.
- If ORION_SLURM_HOST is set, a marked `@pytest.mark.requires_slurm` test submits `hostname` and retrieves stdout.
- Cancelling a long-running local job via the API actually terminates the OS process (verify with ps).
```

### Session 2.4 — Workflow DAG executor

```
Campaigns and agent loops need multi-step workflows. Build a light DAG runner on top of Celery.

Task:
1. backend/common/workflows/ : Workflow = DAG of Job specs with data dependencies ({"uses":"step_id.outputs.energy"}).
2. Pydantic schema for a workflow spec; parse → topological sort → dispatch jobs as predecessors complete (Celery chord/chain or custom state machine reading Job status).
3. Expose POST /api/v1/workflows/ to submit a spec, GET to poll aggregate state.
4. Support fan-out (one parent → many children over a parameter sweep: temperatures=[100,200,300]) via a `foreach` field.
5. Artifacts: workflow run_dir aggregates child artifacts; produce a workflow.json manifest mapping step_id → job_id → artifact_uri.

Acceptance tests:
- A workflow that (a) parses a CIF → (b) runs mock_static → (c) computes derived property works end-to-end.
- Cancelling the workflow cancels all pending children and leaves running ones to finish or be cancelled per spec.
- Fan-out of 8 temperatures produces 8 children that all succeed.
```

---

## PHASE 3 — DFT Engine: Quantum Espresso (Week 4)

Mock energies aren't science. Wire a real first-principles engine with correct conventions.

### Session 3.1 — QE input generation

```
backend/common/engines/qe.py currently does not produce a physically correct pw.x input reliably. Fix.

Task:
1. Use pymatgen.io.pwscf or ase.io.espresso as the base; add a thin wrapper that enforces ORION defaults and calibration.
2. Input generation must set:
   - calculation: scf | relax | vc-relax | nscf | bands — based on JobKind.
   - k-point density via Monkhorst-Pack, parameterized by `kpoint_density` (Å⁻¹) — default 1000 kpts·atom (KPPA=1000) for relax, 4000 for static.
   - ecutwfc / ecutrho: read from pseudopotential family recommendations (SSSP efficiency by default). Ship backend/common/engines/qe/pseudo_recommendations.json.
   - smearing: gaussian for insulators, marzari-vanderbilt 0.02 Ry for metals — auto-select by a pre-pass check (composition heuristic or user override).
   - occupations: fixed for insulators, smearing for metals.
   - XC functional: PBE default, PBEsol option, SCAN via libxc.
3. Pseudopotentials: download SSSP efficiency pseudos into `ORION_PSEUDO_DIR` on first use, cache. Provide a CLI `orion pseudos sync`.
4. Convergence thresholds: conv_thr = 1e-8 Ry, etot_conv_thr = 1e-4 Ry/atom, forc_conv_thr = 1e-3 Ry/Å for relax.
5. Magnetism: read initial moments from a composition-based heuristic (Fe,Co,Ni,Mn,Cr → ferromagnetic start). Expose `magnetic_config` override in inputs.

Acceptance tests:
- For NaCl, Si, Fe (bcc), Cu (fcc): generated pw.in passes `pw.x -inp` parse (dry-run: `pw.x -inp x.in -print` or a checker).
- k-point meshes: NaCl 8×8×8 at KPPA=1000 within ±1 per axis; Si gets same.
- Smearing auto-select: Fe → mv, Si → fixed/gaussian.
```

### Session 3.2 — Run QE and parse outputs

```
Task:
1. Runner: worker copies pseudos + pw.in into run_dir, invokes `mpirun -np N pw.x -inp pw.in > pw.out` via the ExecutionBackend (npool/ntg/ndiag derived from N and kpoints).
2. Parser: pymatgen.io.pwscf.PWOutput is limited; use a combination of ase.io.espresso.read_espresso_out + regex extraction for:
   - Final total energy (Ry → eV), per-atom energy, pressure tensor, forces, stress tensor.
   - SCF convergence trace (n_iter, delta_energy, delta_charge).
   - Relax trajectory: positions at each ionic step.
   - Warning/error flags: "SCF correction compared to forces is too large", "job not converged", "eigenvalues not converged".
3. Persist to Property rows:
   - formation_energy_per_atom: computed as (E_total - Σ μ_i n_i) / N_atoms, where μ_i come from a reference table seeded in DB (elemental ground states computed at same functional + pseudo family). First run: use MP's elemental references as stopgap, flag as "reference:MP_approx".
   - total_magnetization, pressure, stress eigenvalues.
   - bandgap (if nscf+bands run) — zero if metal (E_F within a band).
4. If SCF didn't converge, mark Job failed with reason=scf_not_converged and store diagnostic traces in outputs_json.

Acceptance tests:
- Regression: Si in a supercell, PBE, KPPA=1000, ecutwfc=40 Ry. Bandgap parsed: 0.6 ± 0.1 eV (PBE underestimate, known).
- Al fcc: pressure at relaxed cell ≤ 0.5 kbar.
- A deliberately under-converged run (ecutwfc=10) is flagged as not converged and failed cleanly.
```

### Session 3.3 — QE workflows: relax → static → DOS → bands

```
Task:
1. Compose reusable workflow templates in backend/common/workflows/templates/qe/:
   - relax_then_static: vc-relax → scf at relaxed geometry.
   - band_structure: relax → scf → bands along high-symmetry path (from pymatgen HighSymmKpath).
   - dos: relax → scf (dense k) → dos.x with delta_e=0.01 eV.
   - phonons_gamma: relax → scf → ph.x at Γ only (real phonons in Phase 8+).
2. Expose POST /api/v1/workflows/templates/qe/relax_then_static?structure_id=... which constructs a DAG and submits it.
3. Artifacts: store band_structure.json (kpath + eigenvalues) and dos.csv in MinIO; parse and persist bandgap, VBM, CBM.

Acceptance tests:
- Relax + static on Si produces: a_opt within 0.5% of 5.43 Å; bandgap (indirect) in 0.5–0.8 eV.
- Band structure JSON loadable by pymatgen.BandStructure without errors.
- DOS integrates to correct electron count within 2%.
```

### Session 3.4 — QE calibration + reference energies

```
Formation energies require consistent elemental references at THIS functional + THIS pseudo family, or cross-comparisons are meaningless.

Task:
1. Create a calibration workflow: for each element present in seeded data (Session 1.5), run vc-relax + scf at the fixed ORION defaults and store E_elem_ref (eV/atom) in a ReferenceEnergy table keyed by (element, functional, pseudo_family).
2. Update the formation-energy parser in Session 3.2 to pull from this table. Raise if a reference is missing.
3. CLI `orion calibrate --functional PBE --pseudos SSSP_eff` to trigger the calibration workflow.
4. Cross-validate: formation energies for 10 seeded MP structures computed here vs MP values. Expect agreement within 0.1 eV/atom for PBE; log differences and note any outliers.

Acceptance tests:
- After calibration, `SELECT * FROM reference_energies WHERE functional='PBE'` has ≥10 rows.
- Formation energies for the 10 MP structures: MAE vs MP < 0.15 eV/atom, no single deviation > 0.3 eV/atom.
- CLI is idempotent: re-running skips already-computed elements unless --force.
```

---

## PHASE 4 — MD Engine: LAMMPS (Week 5)

### Session 4.1 — LAMMPS input generation + forcefield registry

```
Task:
1. backend/common/engines/lammps.py: wrap pymatgen.io.lammps.data.LammpsData + a Jinja template for `in.lammps` files.
2. Forcefield registry backend/common/engines/lammps/forcefields/:
   - Lennard-Jones (parametric, for toy systems)
   - EAM (Cu, Ni, Al — ship .alloy files from NIST interatomic potentials repository, with licensing notes)
   - Tersoff (Si, C)
   - ReaxFF (require external install; feature-flag)
   - MACE / NequIP (ML potential — Phase 8)
3. Each forcefield spec declares: applicable_elements, cutoff_Å, timestep_fs_recommended, units, citation.
4. Auto-select: given a composition, search the registry and pick the first compatible forcefield, or fail with "no forcefield covers elements {X}".
5. Ensembles: NVE, NVT (Nose-Hoover, Langevin), NPT (Parrinello-Rahman). Expose damping params and defaults: T_damp = 100*timestep, P_damp = 1000*timestep.

Acceptance tests:
- Generate an input for Cu (108 atoms, EAM, NVT 300K, 100 ps, dt=1 fs) that lammps parses (`lmp -in in.lammps -echo screen -var check 1`).
- Forcefield auto-select for Cu picks EAM; for "Cu,H" it fails cleanly (no compatible FF).
- Unit audits: all timesteps in fs internally; converted to LAMMPS "metal" (ps) at generation.
```

### Session 4.2 — LAMMPS runs + trajectory handling

```
Task:
1. Runner via ExecutionBackend (mpirun lmp -in in.lammps).
2. Trajectory: dump custom every N steps to `dump.lammpstrj` (id type x y z vx vy vz). Post-run, convert to the ASE extxyz format AND to a compressed .h5 via h5py for fast slicing.
3. Analyzers (backend/common/engines/lammps/analysis.py):
   - RDF(r) up to r_max = min(box/2, cutoff + 5 Å), bin=0.05 Å.
   - MSD(t) with per-species breakdown. Fit diffusion coefficient D = slope/6 for 3D on the linear Einstein regime (auto-detect via rolling slope stability).
   - VACF + FT → vibrational DOS.
   - Stress autocorrelation → viscosity via Green-Kubo (NVT, sufficient length).
   - Temperature + pressure time series with block-averaged error bars.
4. Persist summary metrics as Property rows with method="MD-LAMMPS-<ff>" and conditions={T,P,ensemble,n_atoms,timestep,duration_ps}.

Acceptance tests:
- Cu NVT 300K 108 atoms 50 ps: mean T within ±5K of target, P fluctuates around zero (NVT at equilibrium cell).
- LJ liquid (ρ*=0.85, T*=1.0): RDF first peak at r*≈1.13; D*>0.
- Si NPT 300K Tersoff: a_0 within 1% of 5.43 Å after 20 ps equilibration.
```

### Session 4.3 — MD workflow templates

```
Task:
1. Templates:
   - equilibrate_nvt_then_nve: 20 ps NVT → 20 ps NVE (production) with RDF/MSD on the NVE leg.
   - melting_curve: sweep T = [T_low, T_high] in steps; detect melting via MSD jump + enthalpy discontinuity.
   - diffusivity_vs_T: Arrhenius fit of D(T) to extract E_a.
   - elastic_constants_via_strain: apply ±ε strains, measure stress, fit C_ij (for solids).
2. Each template produces a Report object (backend/common/reports/) with a standard JSON schema that the frontend can render.

Acceptance tests:
- Cu melting curve (800–1600K, 100K steps) predicts T_m within ±150K of experimental 1358K (EAM is imperfect; set the bound).
- Arrhenius fit on Li in a simple ionic conductor: E_a in a reasonable range with R² > 0.95.
- Elastic constants workflow on Al: C11 within 15% of 108 GPa.
```

---

## PHASE 5 — Continuum & Mesoscale (Week 6)

Current continuum/mesoscale engines return mocks. Implement minimally honest versions or clearly mark as experimental.

### Session 5.1 — FEM continuum solver (linear elasticity + steady heat)

```
Task:
1. backend/common/engines/continuum.py: use `scikit-fem` (pure Python FEM) or `fenicsx` (if in env). Prefer scikit-fem for zero-install friction.
2. Implement:
   - Linear elastic stress/strain given geometry, boundary conditions, and material (E, ν or full C_ij).
   - Steady-state heat conduction: ∇·(k ∇T) = 0 with Dirichlet/Neumann BCs.
3. Geometry input: either a pymatgen-compatible structure (for periodic bulk) OR an explicit mesh (.msh via meshio).
4. Outputs: displacement field, stress tensor field, temperature field — all stored as VTU in MinIO for ParaView.
5. Validation cases:
   - Cantilever beam: max deflection δ = PL³/(3EI); check against analytical.
   - 1D heat conduction with fixed T at ends: linear profile; check mid-point T.

Acceptance tests:
- Cantilever (steel, 1m × 0.1m × 0.1m, 1 kN tip load): δ within 1% of analytical.
- 1D rod heat: mid-T exactly (T_left+T_right)/2 within FEM tolerance.
- Artifacts open in ParaView (verify manually or with a VTK reader in tests).
```

### Session 5.2 — KMC mesoscale (minimum viable)

```
Task:
1. backend/common/engines/mesoscale.py: implement a rejection-free kMC (Gillespie) for defect migration on a lattice.
2. Events: vacancy hop, interstitial hop, pair annihilation — rates = ν₀ * exp(-E_a/k_B T), with ν₀ and E_a per event type from an input catalog.
3. Time evolution tracked in real physical time; output = concentration of species vs t, cluster size distribution.
4. This is explicitly a MVP — document limitations (no long-range elastic interactions, isotropic lattice only).

Acceptance tests:
- Single vacancy on 100×100×100 simple cubic lattice at 600K with E_a=1 eV: MSD vs t linear; D = a²ν₀/6 * exp(-E_a/kT) within 10% of analytical.
- Annihilation run: starting from 1% vacancies + 1% interstitials, concentrations decay to <0.01% within predicted timescale.
```

### Session 5.3 — Coupling (multi-scale handshake)

```
Task:
1. Workflow template `dft_to_md_to_continuum`:
   - DFT: compute elastic tensor (via Session 8 — deferred dependency noted).
   - MD: compute thermal conductivity at target T.
   - Continuum: use these as material params for a thermomechanical FEM simulation.
2. The coupling is batched, not concurrent (each scale completes before the next starts). Document as "sequential multiscale".

Acceptance tests:
- End-to-end run on Si produces a continuum solution whose material parameters trace back (via provenance, Phase 12) to specific DFT/MD jobs.
```

---

## PHASE 6 — ML: Features, Datasets, Baselines (Week 7)

### Session 6.1 — Featurizers

```
Task:
1. backend/common/ml/features.py: wrap matminer featurizers + add GNN-native structure graph builder.
2. Composition featurizers: ElementProperty (Magpie), Stoichiometry, ValenceOrbital.
3. Structure featurizers: SiteStatsFingerprint, OrbitalFieldMatrix, XRDPowderPattern, SOAP (dscribe).
4. Graph builder: radius graph with cutoff 6 Å, node features = one-hot element + Z + electronegativity + period/group, edge features = [distance, inv_distance, Gaussian-expanded basis (8 centers in [0,cutoff])].
5. Cache featurized representations keyed by (structure_hash, featurizer_id, version). Persist embeddings into structures.embedding (pgvector 256-d) by PCA of concatenated composition features as a default embedding.

Acceptance tests:
- Featurizing 100 structures takes <30s on CPU.
- Round-trip: identical structures → identical graphs (node ordering canonicalized by species+coord lexsort).
- pgvector similarity search: query with Si → nearest neighbors include other group-IV elements / diamond-structure compounds.
```

### Session 6.2 — Dataset registry + splits

```
Task:
1. backend/common/ml/datasets.py: a Dataset is a named, versioned selection of (structure, property, method, conditions) with a deterministic split.
2. Split strategies: random, stratified by composition prototype, structure-cluster (greedy k-center using fingerprint distances — forces extrapolation test), scaffold-like for organics (optional).
3. Snapshot: freeze the row IDs + a content hash; re-running the same dataset version returns identical splits.
4. CLI: `orion dataset create --name oxides_gap_v1 --filter "property=bandgap AND method.functional='PBE'" --split structure-cluster --seed 42`.

Acceptance tests:
- Creating the same dataset twice with the same seed yields the same split hash.
- Structure-cluster split: train/test fingerprint distance distributions differ (KS test p<0.01) — i.e., test is genuinely held-out structures.
```

### Session 6.3 — Baseline models

```
Task:
1. Implement three baselines in backend/common/ml/models/:
   - MeanRegressor (sanity)
   - RandomForest on composition Magpie features (sklearn)
   - XGBoost on composition+structure features
2. Each model has fit / predict / predict_uncertainty (RF: stddev across trees; XGBoost: quantile regression or NGBoost).
3. Unified training script logs metrics (MAE, RMSE, R², spearman) to MLflow (add mlflow to deps; wire MLFLOW_TRACKING_URI).
4. Report per-split performance; save model artifacts to MinIO with a registry row in ml_models.

Acceptance tests:
- On oxides_gap_v1: XGBoost MAE ≤ 0.6 eV (reasonable lower bar for composition-only PBE bandgap).
- Uncertainty coverage: 68% prediction interval covers ≥60% of test values (imperfect is OK; just honest).
- MLflow UI shows three runs with artifacts.
```

### Session 6.4 — GNN (CGCNN-like) training

```
backend/common/ml/models/cgcnn_like.py exists; it needs real training/infra.

Task:
1. PyTorch Lightning wrapper: DataModule reads from the dataset registry, Trainer supports single-GPU with AMP.
2. Loss: Huber for energies/formation, MSE for bandgap, weighted per-target MTL if multi-property.
3. Training config surfaced via YAML; reproducibility enforced (seeds + deterministic cudnn flag + logged lib versions).
4. Uncertainty: evidential regression (NIG output layer) OR deep ensemble of 5 models — pick ensemble for simplicity. Predictions report (μ, σ).
5. POST /api/v1/ml/train spawns a Celery ml-queue task running lightning.fit; logs and checkpoints land in MinIO, registered in ml_models.
6. POST /api/v1/ml/predict(model_id, structure_ids) returns predictions + σ.

Acceptance tests:
- CGCNN on oxides_gap_v1: val MAE ≤ 0.45 eV after 300 epochs on ~200 samples (modest — set by data size).
- Ensemble σ vs |error|: Spearman correlation > 0.2 (non-trivial calibration signal, not perfection).
- Resume training from checkpoint works bit-exact on CPU.
```

### Session 6.5 — Active learning loop

```
Task:
1. backend/common/ml/active_learning.py: given a pool of unlabeled structures + a trained model, query next N by acquisition function.
2. Acquisitions: max σ (uncertainty), UCB (μ + β σ), EI for min/max objectives, BALD (if ensemble).
3. Close the loop via a Celery pipeline: suggest → wait for labels (DFT workflow) → retrain → repeat. Each cycle writes an ALCycle row.
4. API: POST /api/v1/al/campaigns to start a campaign targeting a property+constraint; GET /al/campaigns/{id} shows cycles + cumulative best.

Acceptance tests:
- Simulated AL on oxides_gap_v1: using max-σ, after 10 cycles of 10 queries, the model MAE drops below random-sampling MAE (paired comparison, same compute budget).
- Campaign logs show strictly monotonic cumulative_best_value.
```

---

## PHASE 7 — Agent / BO / Campaigns (Week 8)

### Session 7.1 — Bayesian optimization engine

```
Task:
1. backend/common/ml/bo.py: wrap BoTorch. Support single- and multi-objective (qEHVI for MO).
2. Search spaces: continuous (composition fractions summing to 1), categorical (space group restricted subsets), integer (supercell sizes).
3. Constraints: sum(x_i)=1, charge neutrality (for compositions), formation energy < threshold.
4. Suggest k points per iteration; API: POST /api/v1/bo/suggest with {space, objectives, history}.

Acceptance tests:
- Branin synthetic minimization: reaches ≤0.5 absolute distance from the global min within 30 iterations (budget=5 init + 25 suggestions).
- MO synthetic (ZDT2): Pareto front IGD decreases monotonically over iterations.
```

### Session 7.2 — Campaign orchestrator

```
Task:
1. src/api/routers/campaigns.py + backend/common/campaigns/: a Campaign is {space, budget, objectives, scorer, halting_criteria}.
2. Scorer is a function from structure → score; for materials it composes (ML prediction → DFT confirmation if σ > threshold → score from DFT).
3. Halt on: budget exhausted OR no improvement for K iterations OR target reached.
4. Each campaign step produces a Job and an AgentStep row; the frontend can show the timeline.
5. Resume: a paused campaign resumes from its last completed step.

Acceptance tests:
- A toy campaign optimizing a closed-form synthetic function completes within budget and logs every step.
- Killing the worker mid-campaign and restarting resumes without double-counting.
```

### Session 7.3 — Agent loop (LLM-driven steering, optional but scoped)

```
This is where the "LLM Research Platform" name earns itself.

Task:
1. src/api/routers/agent.py: replace the TODO with a real loop that plans campaign steps using an LLM.
2. The agent is given: current results, remaining budget, objectives, tool catalog (structure_generator, run_dft, run_ml_predict, suggest_bo). It outputs a structured plan (JSON) of next actions.
3. Use Anthropic SDK with claude-opus-4-7. Enforce:
   - System prompt encodes ORION capabilities + unit conventions + cost budget awareness.
   - Tool calls validated against tool schemas before execution.
   - Every LLM call logged (AgentLog table): messages, tool_calls, token_usage, latency.
4. Safety rails: max tools per plan, max total wall-clock per campaign, cost guard ($ estimate per LLM call and per DFT job).
5. Kill switch: POST /api/v1/agent/campaigns/{id}/stop immediately halts the loop.

Acceptance tests:
- Dry-run mode: agent produces a plan for "find oxide with bandgap ~2 eV" that references valid tool names and plausible structures.
- Cost guard: configuring max_cost_usd=0 halts before the first DFT submission.
- Every plan is stored; replaying logs reproduces the same sequence modulo LLM nondeterminism (flag temperature=0 for replay tests).
```

---

## PHASE 8 — Scientific Depth: Elastic, Phonons, Defects (Week 9)

### Session 8.1 — Elastic tensor

```
Task:
1. Workflow: 6-strain scheme (±1%, ±2% isotropic + shear) + relax + compute stress → fit C_ij (6×6 Voigt).
2. Derive bulk modulus (Voigt/Reuss/Hill), shear modulus, Young's, Poisson, elastic anisotropy.
3. Store as structured property with full 6x6 tensor + derived scalars.
4. Stability check: eigenvalues of C > 0 (Born stability). Flag unstable compounds.

Acceptance tests:
- Al (PBE): B_H within 10% of 78 GPa.
- Si: C11/C12/C44 within 10% of 165/64/79 GPa.
- A known unstable structure (e.g., fictitious high-symmetry polymorph) flagged unstable.
```

### Session 8.2 — Phonons (finite displacements via phonopy)

```
Task:
1. Pipeline: use phonopy (`pip install phonopy`) to generate displaced supercells → run QE relax/scf on each → parse forces → compute phonon band + DOS + thermodynamics.
2. Artifacts: phonon_band.json, phonon_dos.csv, thermo.csv (S, C_v, free energy vs T).
3. Imaginary frequencies → mark dynamical instability.

Acceptance tests:
- Si: LO/TO at Γ within 3% of experiment (~15.5 THz).
- Cu: Debye temperature within 10% of 343 K.
- A known unstable structure (cubic BaTiO3 at 0 K) shows imaginary modes.
```

### Session 8.3 — Point defects

```
Task:
1. Workflow: dilute defect in supercell (≥64 atoms), charge states {-2,-1,0,+1,+2}, chemical potential boundaries from phase diagram (pymatgen PhaseDiagram on seeded data).
2. Formation energy E_f(q, μ, E_F) with standard image charge corrections (Freysoldt scheme via pymatgen).
3. Transition levels ε(q/q').

Acceptance tests:
- Si self-interstitial / vacancy energies within ~0.5 eV of published PBE values.
- Charge state cross-over plot rendered from stored data.
```

---

## PHASE 9 — Frontend End-to-End Wiring (Week 10)

The frontend renders but isn't certified end-to-end against real backend data.

### Session 9.1 — API client + auth

```
Task:
1. frontend/src/lib/api.ts: single axios instance with JWT refresh, error interceptor, typed endpoints generated from FastAPI's /openapi.json via `openapi-typescript`.
2. Login/logout/refresh flows; token in httpOnly cookie (adjust backend to set it).
3. Route guards: /dashboard requires auth; /structures, /campaigns, /jobs pages gated by role.

Acceptance tests:
- Playwright: log in as scientist@orion.dev → land on dashboard → refresh page → still authenticated.
- Expired token → silent refresh → request succeeds.
```

### Session 9.2 — Structures UI

```
Task:
1. /structures page: list, search, filter by formula/spacegroup/density.
2. Upload CIF → preview 3D with 3Dmol.js → submit to /structures/parse → redirect to detail page.
3. Structure detail: lattice params, symmetry, composition chart, derived properties table (grouped by method/conditions).
4. Export button with format selector (CIF, POSCAR, XYZ).

Acceptance tests:
- Uploading Si.cif → detail page renders with a=5.43 Å, spacegroup 227, 3D viewer shows 8 atoms in conventional cell.
- Export CIF round-trips (parse the download → matches original).
```

### Session 9.3 — Jobs + workflows UI

```
Task:
1. /jobs page: live list with SSE updates, filter by kind/state, cancel button.
2. Job detail: tabs for Inputs, Outputs, Logs (tail via SSE), Artifacts (download links), Provenance (graph of parents/children).
3. Workflow detail: DAG visualization (reactflow) with per-node status.

Acceptance tests:
- Submit a mock workflow → DAG nodes update pending→running→succeeded live.
- Log tail shows stdout from worker within 2s of emission.
```

### Session 9.4 — Campaigns + ML UI

```
Task:
1. /campaigns: create form for BO/AL campaigns with space/objectives editor.
2. Live progress: best-so-far plot, pareto front (MO), query history table.
3. /ml: model registry browser, train form, predict form with drag-drop structures.

Acceptance tests:
- Create a toy campaign → best-so-far plot updates per cycle.
- ML predict: drop 3 CIFs → returns (μ, σ) per structure within the UI.
```

---

## PHASE 10 — Observability & Performance (Week 11)

### Session 10.1 — Structured logging + tracing

```
Task:
1. structlog everywhere; every request/job has trace_id propagated via OpenTelemetry.
2. FastAPI middleware adds trace_id to response header; Celery tasks inherit via headers.
3. Export OTLP to Jaeger (already in compose); verify spans for: http request → auth → router → db queries → celery submit → worker task → engine subprocess.

Acceptance tests:
- A single /jobs POST appears in Jaeger as one trace spanning API + worker.
- Log lines are JSON with {timestamp, level, trace_id, span_id, event, ...}.
```

### Session 10.2 — Metrics

```
Task:
1. Prometheus counters: requests_total{route,status}, jobs_started_total{kind}, jobs_completed_total{kind,state}, engine_runtime_seconds{engine,kind}.
2. Histograms: http_request_duration_seconds, job_duration_seconds.
3. Gauges: celery_queue_depth, active_workers, db_pool_in_use.
4. Grafana dashboards in docker/grafana/dashboards/: API SLO, Jobs throughput, Engine performance.

Acceptance tests:
- After a load run (50 mock jobs), dashboards show non-empty panels for all four sections.
- SLO dashboard burn-rate panel shows current 1h burn.
```

### Session 10.3 — Load test + perf baselines

```
Task:
1. k6 or locust scripts in tests/perf/: login, list structures, submit jobs, poll jobs.
2. Baseline targets (on a dev box):
   - p95 /materials list < 150 ms at 50 rps.
   - Mock job throughput ≥ 20/s sustained.
   - No memory leak: 1h soak → RSS slope ≤ 5 MB/hr.
3. Publish baselines in docs/perf.md; CI smoke-perf job enforces a relaxed subset.

Acceptance tests:
- Baseline numbers recorded; regressions in CI fail the perf smoke.
```

---

## PHASE 11 — Security hardening (Week 12)

### Session 11.1 — AuthN/Z depth

```
Task:
1. Roles: admin, scientist, viewer. Permissions matrix in backend/common/security/rbac.py.
2. Apply via FastAPI dependency `require_role(...)`. Audit every router.
3. Per-object ownership: a scientist can read any structure but only mutate their own (or their team's).
4. API keys (separate from user JWT) for programmatic access; scoped to specific routes; revocable; hashed at rest.

Acceptance tests:
- Viewer cannot POST to /structures → 403.
- Scientist A cannot delete Scientist B's material → 403.
- API key with scope=read:structures cannot call /jobs → 403.
```

### Session 11.2 — Input validation + rate limits + file handling

```
Task:
1. All user-provided file uploads: size cap (100 MB default), MIME sniffing, magic-byte validation, parsed in a sandboxed subprocess (resource limits via `resource.setrlimit`) to contain malicious CIFs or crafted uploads.
2. Per-user rate limits (slowapi with user-key, not IP): 60 rpm default; engine submissions stricter (10/hr for DFT).
3. SQL injection audit via sqlmap dry-run on login + search endpoints.
4. XSS audit of frontend — ensure no `dangerouslySetInnerHTML` on user content; sanitize with DOMPurify.

Acceptance tests:
- A 500 MB upload is rejected at the edge before disk write.
- 100 logins/s from one user triggers 429.
- sqlmap returns no injectable params.
```

### Session 11.3 — Secret manager integration

```
Task:
1. Abstract secret backend: Local (from env), AWS Secrets Manager, HashiCorp Vault.
2. At startup, read secrets via the chosen backend. Never cache on disk.
3. Rotation: signed URLs for MinIO have TTL ≤ 1 hour; JWT signing keys rotatable via /admin/rotate-key.

Acceptance tests:
- With ORION_SECRETS=aws and a fake AWS Secrets Manager (moto), app starts and reads JWT_SECRET from it.
- Rotating JWT key invalidates previous tokens within 2 minutes.
```

---

## PHASE 12 — Provenance, Reproducibility, Compliance (Week 13)

### Session 12.1 — Full provenance graph

```
Task:
1. Every derived artifact (property, model, prediction, campaign result) carries provenance: input_hashes (structure, code, forcefield/pseudo, config), software_versions (qe, lammps, python, torch), random_seeds, execution_env (image_tag, cpu_model, gpu_model).
2. Store as a Provenance node linked to the artifact; query-able via GET /api/v1/provenance/{id} → full ancestry DAG.
3. Frontend: provenance tab on every detail page renders the DAG.

Acceptance tests:
- A bandgap property row → provenance chain → DFT job → structure → CIF upload → user — all hops traversable.
- Two runs with identical inputs produce byte-identical provenance_hash (modulo wall clock fields).
```

### Session 12.2 — Reproducibility tests

```
Task:
1. Integration tests that lock a full pipeline: CIF → parse → DFT static (single-core, fixed seed where applicable) → bandgap. Assert bandgap reproducible to 1 meV across reruns on the same image tag.
2. Dockerfile pinning: all deps pinned (pip-compile lockfile), image tags pinned, pseudo files SHA256-verified.
3. `orion reproduce <job_id>` CLI: fetches provenance, recreates inputs, reruns, diffs outputs.

Acceptance tests:
- Running `orion reproduce <known_job>` on the same image yields identical outputs to within tolerance.
- Cross-image runs (major image bump) produce a diff report instead of silent mismatch.
```

### Session 12.3 — Data export + licensing

```
Task:
1. Dataset export (CSV/Parquet/CIF bundle) with a manifest: licenses per source, citation BibTeX, schema version.
2. Block exports if any row sourced from a non-redistributable database (flag per source in the DB).
3. GDPR: `orion user forget <email>` drops the user and redacts their personal fields; preserves scientific artifacts under an "anonymous" user.

Acceptance tests:
- Export of a 1k-row dataset produces a zip with manifest.json, data.parquet, citations.bib, LICENSE.txt.
- A dataset containing a row flagged "NO_REDIST" is refused with a clear error.
```

---

## PHASE 13 — Deployment & Production Readiness (Week 14)

### Session 13.1 — Kubernetes manifests

```
Task:
1. Flesh out k8s/base/ with kustomize overlays (dev/staging/prod): Deployment, Service, Ingress, HPA, PDB for api, worker-default, worker-dft, worker-ml, flower. StatefulSets for postgres, redis, minio (prod should use managed — leave helm values for RDS/Elasticache/S3).
2. Helm chart OR kustomize — pick kustomize for simplicity. Overlays differ by replicas, resource requests/limits, ingress hosts.
3. Secrets via SealedSecrets or External-Secrets Operator (document, don't require both).

Acceptance tests:
- `kubectl apply -k k8s/overlays/dev` on a kind cluster brings the stack up. `curl ingress/healthz` returns 200.
- `kubectl drain` on a worker node gracefully drains (preStop hook sends SIGTERM, worker finishes its task, then exits).
```

### Session 13.2 — CI/CD end-to-end

```
Task:
1. GH Actions stages: lint → unit tests → build image (push to ghcr) → integration tests in ephemeral stack (docker-compose + e2e tests) → on main: deploy to dev via ArgoCD trigger or kubectl apply.
2. Image tags: git-sha + semver on tag. Provenance via SLSA-like attestations (cosign sign).
3. DB migrations: run alembic upgrade as a K8s Job before app rollout.

Acceptance tests:
- A PR to main runs the full pipeline in <20 minutes.
- A failing migration blocks rollout (Job fails → Deployment doesn't update).
```

### Session 13.3 — Backups, DR, runbooks

```
Task:
1. Postgres WAL archiving to S3; nightly logical dumps (pg_dump) with 30-day retention.
2. MinIO cross-region replication to a second bucket (or S3 Replication).
3. Runbooks in docs/runbooks/: "DB is down", "Worker queue backed up", "MinIO full", "Bandgaps all wrong after deploy" (data-science incident).
4. Quarterly restore drill scripted (scripts/dr_drill.sh): restore to a shadow stack, run smoke tests, diff.

Acceptance tests:
- Scripted DR drill on a staging clone completes <1 hr to green smoke tests.
- Runbooks pass a dry-run review (each step is literally executable).
```

### Session 13.4 — Docs finalize + release 1.0.0

```
Task:
1. docs/ complete: architecture, data model, engine reference, API reference (auto from OpenAPI), scientific conventions (units, references, pseudos), user guide.
2. CHANGELOG.md curated for 1.0.0 from git log.
3. Tag v1.0.0; GitHub Release with images, changelog, migration notes from 0.x.
4. Post-mortem doc: what was originally stubbed, what the Phase 0→13 sequence fixed, remaining known limitations.

Acceptance tests:
- A new contributor can go from `git clone` to a passing `make verify-all` in <1 hr on a fresh Mac/Linux box, using only docs.
- `pytest` green across unit + integration (slow excluded).
- Release binaries/images published, signed, and pulled successfully.
```

---

## Scientific correctness: global checklist (applies to every phase)

- **Units:** SI internal, explicit conversions at boundaries. `backend/common/units.py` is the only place unit math happens.
- **Energies:** always per-atom AND per-formula-unit versions stored; label which functional and pseudo family.
- **Composition:** reduced formulas canonical; store original stoichiometry separately when relevant.
- **Symmetry:** symprec=0.01 default; expose override; never silently "re-symmetrize" user structures.
- **k-points / cutoffs:** density-based (KPPA, ecutwfc from pseudo recommendation) — never hard-coded integer meshes except in calibration.
- **Thermodynamics:** formation energies against consistent elemental references computed at the SAME functional+pseudo set.
- **Uncertainty:** every ML prediction reports σ; every MD mean reports block-averaged SEM; every DFT value notes method + pseudo.
- **References:** every computed or imported property carries a citation or method fingerprint.
- **No silent interpolation:** if a requested property doesn't exist for a structure, return 404 — never fabricate.

## How to run this plan

1. Start with Phase 0 — it has no science but unblocks everything.
2. Run phases in order; 4 and 5 can swap if you have DFT expertise but no FEM need, but don't skip calibration (3.4).
3. Inside a phase, sessions can parallelize if they don't touch the same files — use the session list to decide.
4. After each phase, write a 1-page "phase report" to docs/history/phase_N_report.md summarizing what landed, what slipped, and what changed scientifically.
5. Don't rewrite this file as you go — append a PROGRESS.md instead. Roadmaps rot when edited live.
