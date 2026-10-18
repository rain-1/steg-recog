"""Run the full generation plan without the pilot-review gate (see scripts/lambda_gen.sh)."""
import subprocess
import sys

from wmscale.common import load_config
from wmscale.pipeline import generation_commands

config = sys.argv[1]
for command in generation_commands(load_config(config), config, 'generate'):
    subprocess.run(command, check=True)
