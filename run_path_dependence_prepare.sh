#!/usr/bin/env bash
set -euo pipefail

# Local-only dataset transformation: reuses the already generated
# category-routing purple/orange/control variants. No GPU datagen.
modal run modal_app.py::path_dependence_prepare \
  --category-data "${CATEGORY_DATA:-runs/category_routing_data}" \
  --output "${OUTPUT:-runs/path_dependence_data}" \
  --warm-per-category "${WARM_PER_CATEGORY:-4}" \
  --main-per-category "${MAIN_PER_CATEGORY:-16}" \
  --val-per-category "${VAL_PER_CATEGORY:-4}" \
  --seed "${DATA_SEED:-15031}"
