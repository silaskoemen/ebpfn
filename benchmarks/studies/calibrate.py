"""Hydra entry point for the shape-matched apparent-SNR calibration gate."""

from pathlib import Path
from typing import Any, cast

import hydra
from benchmarks.studies.apparent_snr_calibration import write_calibration_artifacts
from ebpfn.config import TuningStudyConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../../configs", config_name="tuning")
def main(raw_config: DictConfig) -> None:
    resolved = OmegaConf.to_container(raw_config, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("resolved Hydra configuration must be a mapping")
    config = TuningStudyConfig.model_validate(cast(dict[str, Any], resolved))
    write_calibration_artifacts(config, Path(get_original_cwd()).resolve())


if __name__ == "__main__":
    main()
