r"""XManager launch script for CL-Splats on XCloud.

Usage:
  xmanager launch \
    xcloud/launch.py -- \
    --xm_resource_alloc="group:xcloud/xcloud-shared-user" \
    --output_path=/gcs/xcloud-shared/janackermann/cl-splats-results
"""

from absl import app
from absl import flags
from xmanager import xm
from xmanager import xm_abc

_EXP_NAME = flags.DEFINE_string(
    'exp_name', 'cl-splats', 'Name of the experiment.', short_name='n'
)
_OUTPUT_PATH = flags.DEFINE_string(
    'output_path', None, 'GCS output path for results.'
)
_NUM_GPUS = flags.DEFINE_integer('num_gpus', 1, 'Number of A100 GPUs.')
_DATA_PATH = flags.DEFINE_string(
    'data_path',
    '/gcs/xcloud-shared/janackermann/datasets/cl-splats/Blender-Levels/Level-1',
    'GCS path to dataset. For COLMAP/real-world scenes omit --change_type. '
    'For Blender CL scenes set --change_type (e.g. add, delete, move).',
)
_CHANGE_TYPE = flags.DEFINE_string(
    'change_type',
    'add',
    'Change type for Blender CL dataset (e.g. add, delete, move). '
    'Leave empty for COLMAP/real-world scenes.',
)

flags.mark_flags_as_required(['output_path'])


def main(argv) -> None:
    del argv

    with xm_abc.create_experiment(
        experiment_title=_EXP_NAME.value
    ) as experiment:
        job_requirements = xm.JobRequirements(
            a100=_NUM_GPUS.value,
        )

        executor = xm_abc.Gcp(requirements=job_requirements)

        (executable,) = experiment.package([
            xm.dockerfile_container(
                executor_spec=executor.Spec(),
                args={},
                path='..',
                dockerfile='Dockerfile',
            )
        ])

        experiment.add(
            xm.Job(executable, executor),
            args={
                'args': {
                    'output_path': _OUTPUT_PATH.value,
                    'data_path': _DATA_PATH.value,
                    'change_type': _CHANGE_TYPE.value,
                }
            },
        )


if __name__ == '__main__':
    app.run(main)
