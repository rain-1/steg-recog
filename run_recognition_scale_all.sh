#!/usr/bin/env bash
set -euo pipefail
./run_recognition_scale_datagen.sh
./run_recognition_scale_train.sh
./run_recognition_scale_verbalization.sh
