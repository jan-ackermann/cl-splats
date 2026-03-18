"""Entrypoint for running CL-Splats on XCloud.

This script bridges the XManager launch args with the cl-splats-train CLI.
It handles setting environment variables for offline mode (XCloud containers 
can't access the Internet) and passing the correct arguments.
"""

import os
import subprocess
import logging
from absl import app
from absl import flags

_OUTPUT_PATH = flags.DEFINE_string(
    'output_path', None, 'GCS path for saving results.'
)
_DATA_PATH = flags.DEFINE_string(
    'data_path',
    '/gcs/xcloud-shared/janackermann/datasets/cl-splats/Blender-Levels/Level-1',
    'Path to pre-downloaded dataset on GCS.',
)
_CHANGE_TYPE = flags.DEFINE_string(
    'change_type', 'add', 'Change type for Blender dataset.'
)

flags.mark_flags_as_required(['output_path'])


def main(_):
    # /tmp is always writable; put wandb run data there regardless of mode.
    os.environ['WANDB_DIR'] = '/tmp'

    # Prepare command — data_path is the base scene root (e.g. Level-1).
    # train.py handles appending change_type internally for t > start_time.
    # --offline tells train.py to set HF/torch cache env vars and disable wandb sync.
    cmd_parts = [
        'cl-splats-train',
        f'--data-path {_DATA_PATH.value}',
        f'--change-type {_CHANGE_TYPE.value}',
        '--offline',
    ]

    cmd = ' \\\n    '.join(cmd_parts)
    logging.info('Running command:\n%s', cmd)

    result = subprocess.run(cmd, shell=True)
    exit_code = result.returncode
    if exit_code != 0:
        logging.error('Command failed with exit code %d', exit_code)
        raise RuntimeError(f'Command failed with exit code {exit_code}')

    # Copy offline wandb logs to GCS output path
    logging.info('Copying wandb logs to GCS output path...')
    os.system(f'gsutil -m cp -r /tmp/wandb/* {_OUTPUT_PATH.value}/wandb/ || true')


if __name__ == '__main__':
    app.run(main)
