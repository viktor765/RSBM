"""Claim a run directory for a fresh run, or continue a resumed one in its own.

Hydra creates the directory and its log file before train.py starts, and
writes no record itself (hydra.output_subdir is null), so a refused launch
leaves the run already there untouched.
"""

import json
import shutil
from pathlib import Path

from accelerate.utils import gather_object
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from .models import WEIGHTS, final_model_file_name


def check_launch(cfg) -> None:
    """Refuse an occupied directory, unless resuming a checkpoint inside it.

    Every process checks before any of them writes. All fail if one does,
    and all must have resolved the same directory.
    """
    run_dir = Path(cfg.out_dir)
    try:
        _check(run_dir, f"{HydraConfig.get().job.name}.log", cfg.training.resume.dir)
        error = None
    except (FileExistsError, ValueError) as exc:
        error = exc
    results = gather_object([(str(run_dir.resolve()), error)])
    if len({directory for directory, _ in results}) > 1:
        raise RuntimeError(
            "Processes resolved different run directories; set hydra.run.dir."
        )
    for _, error in results:
        if error is not None:
            raise error


def _check(run_dir: Path, log_name: str, resume_dir) -> None:
    # Hydra's log file is new, and this launch has not yet logged to it.
    earlier = [
        entry.name
        for entry in run_dir.iterdir()
        if entry.name != log_name or entry.stat().st_size > 0
    ]
    if resume_dir is None:
        if earlier:
            raise FileExistsError(
                f"{run_dir} holds an earlier run. To continue it, resume with "
                "training.resume.dir=<its checkpoint>. If this is a different "
                "experiment whose settings map to the same directory, choose "
                "another directory with hydra.run.dir."
            )
        return
    checkpoints = (run_dir / "checkpoints").resolve()
    resume_path = Path(resume_dir).resolve()
    if not resume_path.is_relative_to(checkpoints):
        owner = next(
            (parent.parent for parent in resume_path.parents if parent.name == "checkpoints"),
            None,
        )
        hint = f" (hydra.run.dir={owner})" if owner is not None else ""
        raise ValueError(
            f"A resumed run continues in its checkpoint's run directory{hint}, "
            f"not {run_dir}."
        )
    if not (run_dir / ".hydra" / "config.yaml").is_file():
        raise ValueError(f"{run_dir} has no launch record to resume.")


def record_launch(cfg, revision: dict) -> Path:
    """Save this launch's config, Hydra settings, overrides and git revision.

    The first launch's record is `.hydra/` and `git.json`, which later
    launches keep; resumed launches add theirs under `.hydra/resumes/<n>/`.
    Returns the record's directory.
    """
    run_dir = Path(cfg.out_dir)
    record = run_dir / ".hydra"
    git_path = run_dir / "git.json"
    if cfg.training.resume.dir is not None:
        resumes = record / "resumes"
        previous = [int(p.name) for p in resumes.iterdir()] if resumes.is_dir() else []
        record = resumes / str(max(previous, default=0) + 1)
        git_path = record / "git.json"
    record.mkdir(parents=True)
    hydra_cfg = HydraConfig.get()
    OmegaConf.save(cfg, record / "config.yaml", resolve=True)
    OmegaConf.save(OmegaConf.create({"hydra": hydra_cfg}), record / "hydra.yaml")
    OmegaConf.save(hydra_cfg.overrides.task, record / "overrides.yaml")
    git_path.write_text(json.dumps(revision, indent=2) + "\n")
    return record


def archive_abandoned_history(run_dir, step, *, include_step, resumed_from) -> None:
    """Set aside what a run recorded after the checkpoint it resumes from.

    Resuming the checkpoint at `step` abandons the continuation that followed
    it. Its curve rows and evaluation directories (at `step` too when
    `include_step`, because that evaluation follows the checkpoint) move to
    `eval/abandoned/<n>/`, with copies of the original curve and diagnostic
    files. Diagnostic windows ending after `step` leave the active file; those
    ending at it were written before the checkpoint. Later numbered
    checkpoints, `c_final` and the final exports move to
    `checkpoints/abandoned/<n>/`; the resumed run writes its own final ones.
    Earlier records stay unchanged, so the active records are those of an
    uninterrupted run.
    """
    root = Path(run_dir)
    eval_dir, checkpoint_dir = root / "eval", root / "checkpoints"

    def evaluated_later(record_step):
        return record_step > step or (include_step and record_step == step)

    records = {
        eval_dir / "curve.jsonl": evaluated_later,
        root / "training_diagnostics.jsonl": lambda record_step: record_step > step,
    }
    kept = {}
    for path, abandoned in records.items():
        if path.exists():
            rows = path.read_text().splitlines(keepends=True)
            keep = [row for row in rows if not abandoned(json.loads(row)["step"])]
            if len(keep) < len(rows):
                kept[path] = keep
    step_dirs = [
        path
        for path in (eval_dir.iterdir() if eval_dir.is_dir() else ())
        if path.is_dir() and path.name.isdigit() and evaluated_later(int(path.name))
    ]
    resumed_from = Path(resumed_from).resolve()
    later = [
        path
        for path in checkpoint_dir.glob("c_*")
        if path.is_dir()
        and not path.is_symlink()
        and path.resolve() != resumed_from
        and (
            path.name == "c_final"
            or (path.name[2:].isdigit() and int(path.name[2:]) > step)
        )
    ]
    for weights in WEIGHTS:
        if (checkpoint_dir / final_model_file_name(weights)).exists():
            later.append(checkpoint_dir / final_model_file_name(weights))
    if not (kept or step_dirs or later):
        return
    index = 1
    while (eval_dir / "abandoned" / str(index)).exists() or (
        checkpoint_dir / "abandoned" / str(index)
    ).exists():
        index += 1
    archive = eval_dir / "abandoned" / str(index)
    archive.mkdir(parents=True)
    for path in records:
        if path.exists():
            shutil.copy2(path, archive / path.name)
    for path, keep in kept.items():
        temporary = path.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(keep))
        temporary.replace(path)
    for path in step_dirs:
        shutil.move(str(path), archive / path.name)
    if later:
        destination = checkpoint_dir / "abandoned" / str(index)
        destination.mkdir(parents=True)
        for path in later:
            shutil.move(str(path), destination / path.name)
    (archive / "resume.json").write_text(json.dumps({"checkpoint_step": step}))
