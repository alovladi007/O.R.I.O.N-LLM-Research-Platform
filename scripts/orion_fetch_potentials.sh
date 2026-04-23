#!/usr/bin/env bash
# Pull the EAM interatomic potentials referenced by ORION's LAMMPS
# forcefield registry (Session 4.1) into
# backend/common/engines/lammps_input/forcefields/data/.
#
# The files are not bundled in-tree because they are third-party data,
# and we'd rather the registry fall back to LJ than ship bytes whose
# provenance users can't verify from this repo's history alone.
#
# Run once after cloning if you want EAM runs out of the box:
#   bash scripts/orion_fetch_potentials.sh
#
# Sources: NIST Interatomic Potentials Repository (public domain under
# 17 USC §105). See forcefields/data/LICENSE.txt for the full note.

set -euo pipefail

DEST="$(cd "$(dirname "$0")/.." && pwd)/backend/common/engines/lammps_input/forcefields/data"
mkdir -p "$DEST"

urls=(
  # Cu: Foiles/Baskes/Daw 1986
  "https://www.ctcms.nist.gov/potentials/Download/1986--Foiles-S-M-Baskes-M-I-Daw-M-S--Cu/1/Cu_u3.eam"
  # Ni + Al: Mishin et al. 1999
  "https://www.ctcms.nist.gov/potentials/Download/1999--Mishin-Y-Farkas-D-Mehl-M-J-Papaconstantopoulos-D-A--Ni-Al/1/Ni99_v2.eam.alloy"
  "https://www.ctcms.nist.gov/potentials/Download/1999--Mishin-Y-Farkas-D-Mehl-M-J-Papaconstantopoulos-D-A--Ni-Al/1/Al99_v2.eam.alloy"
)

for u in "${urls[@]}"; do
  name="$(basename "$u")"
  out="$DEST/$name"
  if [[ -f "$out" ]]; then
    echo "[skip] $name already present"
    continue
  fi
  echo "[fetch] $name"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$u" -o "$out"
  else
    wget -q "$u" -O "$out"
  fi
done

echo "Done. Files in $DEST"
