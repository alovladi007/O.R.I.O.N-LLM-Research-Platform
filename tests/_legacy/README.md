# tests/_legacy — pre-refactor tests

Files in this directory are named with `.pre_refactor` suffix so pytest won't
collect them. They depend on:

- `src.api.auth.security` — module not present
- `src.api.models.Simulation` — class not present
- `backend.common.structures.to_cif` / `to_poscar` / `to_xyz` —
  functions not present in the current module

When the corresponding modules land (Sessions 1.1 / 1.2 / 1.4 / 1.5), the
tests here can be revived, adapted, and moved back to `tests/` — or
re-written as new test cases. Don't delete them blind; they encode
expectations from the prior session work that are still useful signal.
