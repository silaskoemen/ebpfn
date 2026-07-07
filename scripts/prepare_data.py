"""Hydra application boundary for preparing an OpenML source manifest."""

import json
from pathlib import Path
from typing import Any
from typing import cast

import hydra
from benchmarks.data import canonical_openml_task
from benchmarks.data import load_openml_source
from ebpfn.config import PrepareDataConfig
from ebpfn.data import build_evaluation_task
from ebpfn.utils import environment_provenance
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from omegaconf import OmegaConf


@hydra.main(version_base=None, config_path="../configs", config_name="prepare_data")
def main(raw_config: DictConfig) -> None:
    resolved = OmegaConf.to_container(raw_config, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("resolved Hydra configuration must be a mapping")
    config = PrepareDataConfig.model_validate(cast(dict[str, Any], resolved))
    project_root = Path(get_original_cwd()).resolve()
    output = (project_root / config.mode.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = load_openml_source(config.openml, config.data.split)
    result = build_evaluation_task(canonical_openml_task(source), source.split, config.data)
    payload = {
        "resolved_config": resolved,
        "source_id": source.source_id,
        "outer_split_id": source.split.outer_split_id,
        "admitted": result.eligibility.admitted,
        "exclusion_reasons": list(result.eligibility.reasons),
    }
    (output / "config.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    (output / "environment.json").write_text(json.dumps(environment_provenance(project_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
