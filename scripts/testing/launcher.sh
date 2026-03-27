#!/bin/bash

# launch each script in succession
SCRIPT_DIR="/mnt/data_sda/jdearborn/scratch/pabpc1-borzoi-analysis/scripts"
CONFIG_PATH="/mnt/data_sda/jdearborn/scratch/pabpc1-borzoi-analysis/manuscript_runs/fig_3.yaml"

# python ${SCRIPT_DIR}/01_get_attributions.py --config ${CONFIG_PATH}
# echo "01_get_attributions.py completed"

# python ${SCRIPT_DIR}/02_run_modisco.py --config ${CONFIG_PATH}
# echo "02_run_modisco.py completed"

python ${SCRIPT_DIR}/03_run_enrichment.py --config ${CONFIG_PATH}
echo "03_run_enrichment.py completed"

# python ${SCRIPT_DIR}/04_map_motifs.py --config ${CONFIG_PATH}
# echo "04_map_motifs.py completed"
