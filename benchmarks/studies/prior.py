"""Hydra entry point for the Step 3 prior p-complexity audit."""

from pathlib import Path
from typing import Any, cast

import hydra
from benchmarks.studies.prior_audit import write_study_artifacts
from ebpfn.config import PriorStudyConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../../configs", config_name="prior")
def main(raw_config: DictConfig) -> None:
    resolved = OmegaConf.to_container(raw_config, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("resolved Hydra configuration must be a mapping")
    config = PriorStudyConfig.model_validate(cast(dict[str, Any], resolved))
    project_root = Path(get_original_cwd()).resolve()
    write_study_artifacts(config, project_root)


if __name__ == "__main__":
    main()
