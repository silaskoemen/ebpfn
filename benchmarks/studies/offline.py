"""Hydra entry point for the Step-5 PFN learning-curve panel."""

from pathlib import Path
from typing import Any, cast

import hydra
from benchmarks.studies.offline_validation import write_training_panel_artifacts
from ebpfn.config import OfflineValidationConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../../configs", config_name="offline_validation")
def main(raw_config: DictConfig) -> None:
    resolved = OmegaConf.to_container(raw_config, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("resolved Hydra configuration must be a mapping")
    config = OfflineValidationConfig.model_validate(cast(dict[str, Any], resolved))
    write_training_panel_artifacts(config, Path(get_original_cwd()).resolve())


if __name__ == "__main__":
    main()
