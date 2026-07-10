"""Hydra entry point for the Step 4 recovery and tuning study."""

from pathlib import Path
from typing import Any, cast

import hydra
from benchmarks.studies.tuning_recovery import write_study_artifacts
from ebpfn.config import TuningStudyConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../../configs", config_name="tuning")
def main(raw_config: DictConfig) -> None:
    resolved = OmegaConf.to_container(raw_config, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("resolved Hydra configuration must be a mapping")
    config = TuningStudyConfig.model_validate(cast(dict[str, Any], resolved))
    project_root = Path(get_original_cwd()).resolve()
    write_study_artifacts(config, project_root)


if __name__ == "__main__":
    main()
