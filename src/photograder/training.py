"""Local-first training with explicit stages and recorded evaluation protocols."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .runtime import (
    atomic_json,
    environment_info,
    file_sha256,
    load_checkpoint,
    save_checkpoint,
    seed_everything,
    select_device,
    synchronize,
)

DEFAULTS = {
    "stage": "global",
    "epochs": 10,
    "batch_size": 2,
    "accumulation_steps": 4,
    "learning_rate": 0.0002,
    "weight_decay": 0.0001,
    "seed": 42,
    "device": "auto",
    "patience": 8,
    "max_steps": None,
    "num_workers": 0,
    "preservation": True,
    "clip_grad_norm": 1.0,
    "log_every": 25,
    "combined_faults_after_micro_step": None,
    "validation": True,
    "cpu_threads": None,
    "pin_memory": None,
    "persistent_workers": False,
    "prefetch_factor": 2,
    "non_blocking": True,
    "max_seconds": None,
}
LOSS_DEFAULTS = {
    "reconstruction": 1.0,
    "lut_smooth": 0.001,
    "lut_monotonic": 0.01,
    "map_smooth": 0.001,
    "preservation": 0.2,
    "identity": 0.1,
    "gamut": 0.01,
    "exposure_aux": 0.0,
}


def masked_mean(value, mask):
    weight = mask.expand_as(value)
    return (value * weight).sum() / weight.sum().clamp_min(1)


def reconstruction(prediction, target, mask):
    return masked_mean((prediction - target).abs(), mask)


def recipe_penalties(recipe):
    """Normalize all penalties by their number of supported samples."""
    tensors = [
        getattr(recipe, name, None)
        for name in ("luts", "exposure", "log_contrast", "rgb_residual", "bilateral_grid")
    ]
    ref = next((value for value in tensors if isinstance(value, torch.Tensor)), None)
    if ref is None and getattr(recipe, "stages", None):
        totals = [recipe_penalties(stage) for stage in recipe.stages]
        return {key: sum(item[key] for item in totals) / len(totals) for key in totals[0]}
    zero = ref.new_zeros(()) if ref is not None else torch.tensor(0.0)
    result = {key: zero for key in ("lut_smooth", "lut_monotonic", "map_smooth")}
    lut = getattr(recipe, "luts", None)
    if lut is not None:
        differences = [lut.diff(dim=axis) for axis in (2, 3, 4)]
        result["lut_smooth"] = sum(d.square().mean() for d in differences) / 3
        result["lut_monotonic"] = sum(F.relu(-d).square().mean() for d in differences) / 3
    maps = [getattr(recipe, name, None) for name in ("exposure", "log_contrast")]
    maps = [m for m in maps if m is not None]
    if maps:
        differences = [m.diff(dim=axis).abs().mean() for m in maps for axis in (-1, -2) if m.shape[axis] > 1]
        result["map_smooth"] = sum(differences) / max(1, len(differences))
    return result


def _analyze(model, image, style, bypass=False, valid_mask=None):
    options = {"valid_mask": valid_mask} if getattr(model.config, "global_context", False) else {}
    return model.analyze(image, style, bypass_global=bypass, **options)


def exposure_auxiliary_loss(recipe, local_input, valid_mask, corruption):
    """Supervise only known exposure inverses using the actual rendered map.

    The caller restricts this to the repair stage. Pure exposure and identity
    rows have exact inverse stops; contrast and combined faults deliberately do
    not. Missing-map architectures and batches with no known inverse return
    zero. No inverse label is inferred from a human-edited target.
    """
    from .renderer import exposure_field

    zero = local_input.new_zeros(())
    if corruption is None or recipe.exposure is None or recipe.stages:
        return zero
    known = corruption["inverse_known"].to(device=local_input.device, dtype=local_input.dtype)
    weights = valid_mask * known[:, None, None, None]
    inverse = corruption["known_inverse"].to(device=local_input.device, dtype=local_input.dtype)
    predicted = exposure_field(local_input, recipe)
    return masked_mean((predicted - inverse).abs(), weights)


def _loss(model, batch, settings, weights, micro_step):
    from .corruptions import corrupt_batch
    from .renderer import render, render_with_aux

    image, target, mask, style = (batch[k] for k in ("image", "target", "valid_mask", "style_id"))
    stage = settings["stage"]
    seed = settings["seed"] * 1000003 + micro_step
    bypass = stage == "repair"
    identity = False
    support = None
    reference = None
    corrupted = None
    threshold = settings.get("combined_faults_after_micro_step")
    allow_combinations = threshold is not None and micro_step >= threshold
    if stage == "repair":
        clean = target
        corrupted = corrupt_batch(
            clean, seed=seed, allow_identity=True, valid_mask=mask, allow_combinations=allow_combinations
        )
        image, support = (corrupted["image"], corrupted["mask"])
        reference = clean
    elif stage == "joint" and micro_step % 4 == 3:
        if micro_step % 16 == 15:
            image = target
            bypass, identity = (True, True)
        else:
            with torch.no_grad():
                clean_recipe = _analyze(model, image, style, valid_mask=mask)
                reference = render(image, clean_recipe).detach()
            corrupted = corrupt_batch(
                image, seed=seed, allow_identity=False, valid_mask=mask, allow_combinations=allow_combinations
            )
            image, support = (corrupted["image"], corrupted["mask"])
    recipe = _analyze(model, image, style, bypass, valid_mask=mask)
    output, auxiliary = render_with_aux(image, recipe, grade_strength=0.0 if bypass else 1.0)
    terms = {"reconstruction": reconstruction(output, target, mask)}
    terms.update(recipe_penalties(recipe))
    if not bypass and hasattr(model, "regularization"):
        regularization = model.regularization()
        terms["lut_smooth"] = regularization.get("lut_smoothness", output.new_zeros(()))
        terms["lut_monotonic"] = regularization.get("lut_monotonicity", output.new_zeros(()))
    pre_gamut = auxiliary.get("pre_gamut", output)
    terms["gamut"] = masked_mean(F.relu(-pre_gamut).square() + F.relu(pre_gamut - 1).square(), mask)
    terms["identity"] = reconstruction(output, image, mask) if identity else output.new_zeros(())
    terms["preservation"] = output.new_zeros(())
    terms["exposure_aux"] = output.new_zeros(())
    if stage == "repair" and weights.get("exposure_aux", 0) > 0:
        terms["exposure_aux"] = exposure_auxiliary_loss(recipe, image, mask, corrupted)
    if settings["preservation"] and support is not None and (reference is not None):
        outside = 1 - F.max_pool2d(support.float(), kernel_size=9, stride=1, padding=4)
        terms["preservation"] = reconstruction(output, reference, outside * mask)
    total = sum((weights.get(name, 0) * value for name, value in terms.items()))
    return (total, {key: value.detach() for key, value in terms.items()})


def _move_batch(batch, device, non_blocking=True):
    asynchronous = bool(non_blocking and device.type == "cuda")
    return {
        key: value.to(device, non_blocking=asynchronous) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _log_components(loss, components):
    names = list(components)
    values = torch.stack([loss.detach(), *(components[name] for name in names)]).float().cpu().tolist()
    return (values[0], dict(zip(names, values[1:], strict=False)))


def _check_finite(loss, device, message):
    if device.type == "cuda":
        torch._assert_async(torch.isfinite(loss), message)
    elif not bool(torch.isfinite(loss)):
        raise RuntimeError(message)


def _loader(dataset, settings, device, *, shuffle, generator):
    workers = settings["num_workers"]
    options = {
        "batch_size": settings["batch_size"],
        "shuffle": shuffle,
        "generator": generator,
        "num_workers": workers,
        "pin_memory": device.type == "cuda" if settings["pin_memory"] is None else settings["pin_memory"],
    }
    if workers:
        options.update(
            persistent_workers=settings["persistent_workers"], prefetch_factor=settings["prefetch_factor"]
        )
    return DataLoader(dataset, **options)


@torch.no_grad()
def validation_loss(model, loader, device, stage, seed, non_blocking=True):
    from .corruptions import corrupt_batch
    from .renderer import render

    model.eval()
    total, count = (torch.zeros((), device=device), 0)
    for index, batch in enumerate(loader):
        batch = _move_batch(batch, device, non_blocking)
        if stage == "repair":
            source = corrupt_batch(
                batch["target"],
                seed=seed + index,
                allow_identity=True,
                valid_mask=batch["valid_mask"],
                allow_combinations=False,
            )["image"]
            recipe = _analyze(model, source, batch["style_id"], True, batch["valid_mask"])
            output = render(source, recipe, grade_strength=0)
        else:
            recipe = _analyze(model, batch["image"], batch["style_id"], valid_mask=batch["valid_mask"])
            output = render(batch["image"], recipe)
        n = batch["image"].shape[0]
        total.add_(reconstruction(output, batch["target"], batch["valid_mask"]).detach() * n)
        count += n
    model.train()
    if not count:
        raise ValueError("Validation split is empty. Supply scene-disjoint validation data.")
    return float((total / count).cpu())


def _resolve(config):
    config = json.loads(json.dumps(config))
    config["training"] = {**DEFAULTS, **config.get("training", {})}
    config["loss"] = {**LOSS_DEFAULTS, **config.get("loss", {})}
    config.setdefault("data", {})
    config["data"].setdefault("image_size", 512)
    config["data"].setdefault("limit", None)
    config.setdefault("model", {})
    config.setdefault("output_dir", "runs/experiment")
    settings = config["training"]
    if settings["stage"] not in ("global", "repair", "joint"):
        raise ValueError("stage must be global, repair, or joint")
    for name in ("epochs", "batch_size", "accumulation_steps"):
        if settings[name] < 1:
            raise ValueError(f"{name} must be positive")
    for name in ("num_workers", "log_every"):
        if type(settings[name]) is not int or settings[name] < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if type(settings["prefetch_factor"]) is not int or settings["prefetch_factor"] < 1:
        raise ValueError("prefetch_factor must be a positive integer")
    if settings["cpu_threads"] is not None and (
        type(settings["cpu_threads"]) is not int or settings["cpu_threads"] < 1
    ):
        raise ValueError("cpu_threads must be a positive integer or null")
    for name in ("validation", "persistent_workers", "non_blocking"):
        if type(settings[name]) is not bool:
            raise ValueError(f"{name} must be boolean")
    if settings["pin_memory"] is not None and type(settings["pin_memory"]) is not bool:
        raise ValueError("pin_memory must be boolean or null")
    if settings["max_seconds"] is not None and (
        not isinstance(settings["max_seconds"], (int, float))
        or not math.isfinite(settings["max_seconds"])
        or settings["max_seconds"] <= 0
    ):
        raise ValueError("max_seconds must be a positive finite number or null")
    if settings["max_steps"] is not None and (
        type(settings["max_steps"]) is not int or settings["max_steps"] < 1
    ):
        raise ValueError("max_steps must be a positive integer or null")
    threshold = settings["combined_faults_after_micro_step"]
    if threshold is not None and (type(threshold) is not int or threshold < 0):
        raise ValueError("combined_faults_after_micro_step must be a nonnegative integer or null")
    aux_weight = config["loss"]["exposure_aux"]
    if not isinstance(aux_weight, (int, float)) or not math.isfinite(aux_weight) or aux_weight < 0:
        raise ValueError("exposure_aux must be a finite nonnegative weight")
    return config


def train(
    config: dict,
    *,
    resume=None,
    initialize=None,
    train_dataset=None,
    validation_dataset=None,
    checkpoint_callback=None,
) -> dict:
    """Train a phase with verified paired data and resumable checkpoints.

    ``validation=False`` selects only the declared schedule's final checkpoint.
    A wall-time safety stop saves a resumable last.pt, never a claimed final
    checkpoint. The budget is per invocation and checked at optimizer boundaries.
    ``checkpoint_callback(output_directory, record)`` runs after durable writes.
    """
    from .corruptions import CORRUPTION_VERSION
    from .data import PairedDataset, read_manifest, validate_manifest
    from .models import ModelConfig, build_model, initialize_from_model

    started = time.perf_counter()
    config = _resolve(config)
    settings = config["training"]
    device = select_device(settings["device"])
    seed_everything(settings["seed"])
    torch.set_num_threads(settings["cpu_threads"] or min(4, torch.get_num_threads()))
    output = Path(config["output_dir"])
    if not resume and any(
        (output / name).exists() for name in ("best.pt", "last.pt", "history.json", "config.json")
    ):
        raise FileExistsError(f"Experiment already exists at {output}; choose a new directory or resume it.")
    output.mkdir(parents=True, exist_ok=True)
    manifest = Path(config["data"]["manifest"])
    manifest_rows = read_manifest(manifest)
    if train_dataset is None:
        manifest_info = validate_manifest(manifest)
    else:
        manifest_info = {"styles": {row["style_id"]: row["style_name"] for row in manifest_rows}}
    data_options = {"image_size": config["data"]["image_size"], "limit": config["data"]["limit"]}
    training_data = (
        train_dataset if train_dataset is not None else PairedDataset(manifest, "train", **data_options)
    )
    val_limit = config["data"].get("val_limit")
    validation_data = None
    if settings["validation"]:
        validation_data = (
            validation_dataset
            if validation_dataset is not None
            else PairedDataset(manifest, "val", image_size=data_options["image_size"], limit=val_limit)
        )
    elif validation_dataset is not None:
        raise ValueError("validation_dataset cannot be supplied when validation=False")
    if not len(training_data) or (settings["validation"] and (not len(validation_data))):
        raise ValueError("Training requires a nonempty train split and, when enabled, a nonempty val split.")
    manifest_digest = file_sha256(manifest)
    for dataset, split, limit in (
        (train_dataset, "train", data_options["limit"]),
        (validation_dataset, "val", val_limit),
    ):
        if dataset is None:
            continue
        expected = [row for row in manifest_rows if row["split"] == split]
        if limit is not None:
            expected = expected[:limit]
        if getattr(dataset, "rows", None) != expected:
            raise ValueError(f"Dataset override rows differ from the requested manifest {split} rows.")
        if getattr(dataset, "manifest_sha256", manifest_digest) != manifest_digest:
            raise ValueError("Dataset override manifest checksum differs.")
        if getattr(dataset, "image_size", data_options["image_size"]) != data_options["image_size"]:
            raise ValueError("Dataset override image_size differs from training configuration.")
    model_config = dict(config["model"])
    if "styles" not in model_config:
        names = manifest_info["styles"]
        if sorted(names) != list(range(len(names))):
            raise ValueError("Manifest style IDs must be contiguous from zero when deriving styles.")
        model_config["styles"] = tuple(names[index] for index in range(len(names)))
    if "styles" in model_config:
        model_config["styles"] = tuple(model_config["styles"])
    for index, name in manifest_info["styles"].items():
        if index >= len(model_config["styles"]) or model_config["styles"][index] != name:
            raise ValueError(f"Manifest style {index}={name!r} does not match model.styles.")
    model = build_model(ModelConfig(**model_config)).to(device)
    initial_epoch, initial_batch, step, best, failures = (0, 0, 0, math.inf, 0)
    backend = type(training_data).__module__ + "." + type(training_data).__name__
    if resume:
        if Path(resume).resolve() != (output / "last.pt").resolve():
            raise ValueError(
                "Resume must use this output directory's last.pt; use --initialize to fork a run."
            )
        restored, payload = load_checkpoint(resume, device)
        if asdict(restored.config) != asdict(model.config):
            raise ValueError("Resume model configuration differs from checkpoint.")
        if payload.get("metadata", {}).get("manifest_sha256") != file_sha256(manifest):
            raise ValueError("Resume manifest differs from the checkpoint's training data.")
        previous = _resolve(payload["training_config"])
        for key in (
            "stage",
            "batch_size",
            "accumulation_steps",
            "learning_rate",
            "weight_decay",
            "seed",
            "preservation",
            "combined_faults_after_micro_step",
            "validation",
        ):
            if previous["training"][key] != settings[key]:
                raise ValueError(f"Cannot change {key} on resume; use --initialize for a new experiment.")
        if (
            settings["stage"] != "global"
            and payload.get("metadata", {}).get("corruption_version") != CORRUPTION_VERSION
        ):
            raise ValueError("Corruption generator changed; start a new experiment instead of resuming.")
        if previous["data"] != config["data"] or previous["loss"] != config["loss"]:
            raise ValueError("Cannot change data or loss settings on resume; use --initialize instead.")
        model = restored
        initial_epoch, step = (payload["epoch"], payload["step"])
        cursor = payload.get("metadata", {}).get("resume_cursor", {})
        initial_epoch = cursor.get("epoch_index", initial_epoch)
        initial_batch = cursor.get("next_batch_index", 0)
        previous_backend = payload.get("metadata", {}).get("data_backend", backend)
        if previous_backend != backend:
            raise ValueError("Cannot change data backend on resume; batch ordering may differ.")
        if initial_epoch >= settings["epochs"] or (
            settings["max_steps"] is not None and step >= settings["max_steps"]
        ):
            raise ValueError(
                "Checkpoint already meets the requested epoch/update budget; increase it to resume."
            )
        best = payload.get("metadata", {}).get("best_val_loss")
        best = math.inf if best is None else best
        if "initialization" in previous:
            config["initialization"] = previous["initialization"]
        if "torch_rng_state" in payload:
            torch.set_rng_state(payload["torch_rng_state"])
        if device.type == "cuda" and "cuda_rng_states" in payload:
            torch.cuda.set_rng_state_all(payload["cuda_rng_states"])
    elif initialize:
        restored, payload = load_checkpoint(initialize, "cpu")
        initialization_report = initialize_from_model(model, restored)
        config["initialization"] = {
            "checkpoint": str(initialize),
            "sha256": file_sha256(initialize),
            **initialization_report,
            "source_config": payload["model_config"],
        }
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    if resume and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    config["model"] = asdict(model.config)
    metadata = {
        "manifest_sha256": file_sha256(manifest),
        "training_samples": len(training_data),
        "validation_samples": len(validation_data) if validation_data is not None else 0,
        "validation_enabled": settings["validation"],
        "trained": False,
        "stage": settings["stage"],
        "selected_by": "validation_l1" if settings["validation"] else "fixed_schedule_final",
        "final_selected": False,
        "data_backend": backend,
        "renderer_dtype": "float32",
        "environment": environment_info(device),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "corruption_version": CORRUPTION_VERSION,
        "repair_validation_protocol": {
            "allow_combinations": False,
            "allow_identity": True,
            "families": ["ellipse", "gradient", "irregular"],
            "seed_rule": "training seed + 900000 + validation batch index",
        }
        if settings["validation"]
        else None,
        "source_sha256": {path.name: file_sha256(path) for path in Path(__file__).parent.glob("*.py")},
        "pilot": len(training_data) < 1000 or bool(config["data"]["limit"]) or settings["epochs"] <= 10,
    }
    if resume:
        metadata.update(best_epoch=payload.get("metadata", {}).get("best_epoch"))
        failures = payload.get("metadata", {}).get("early_stopping_failures", 0)
    atomic_json(output / "config.json", config)
    atomic_json(output / "environment.json", metadata)
    train_generator = torch.Generator()
    loader = _loader(training_data, settings, device, shuffle=True, generator=train_generator)
    val_loader = (
        _loader(
            validation_data,
            settings,
            device,
            shuffle=False,
            generator=torch.Generator().manual_seed(settings["seed"]),
        )
        if validation_data is not None
        else None
    )
    history = (
        json.loads((output / "history.json").read_text())
        if resume and (output / "history.json").exists()
        else []
    )
    model.train()
    total_samples = 0
    stop_reason = None
    final_selected = False

    def budget_expired():
        return (
            settings["max_seconds"] is not None and time.perf_counter() - started >= settings["max_seconds"]
        )

    def persist(*, checkpoint_epoch, improved=False, final=False, reason=None):
        save_checkpoint(
            output / "last.pt",
            model,
            optimizer,
            epoch=checkpoint_epoch,
            step=step,
            training_config=config,
            metadata=metadata,
        )
        if improved or final:
            save_checkpoint(
                output / "best.pt",
                model,
                optimizer,
                epoch=checkpoint_epoch,
                step=step,
                training_config=config,
                metadata=metadata,
            )
        if final:
            save_checkpoint(
                output / "final.pt",
                model,
                optimizer,
                epoch=checkpoint_epoch,
                step=step,
                training_config=config,
                metadata=metadata,
            )
        atomic_json(output / "history.json", history)
        if checkpoint_callback is not None:
            checkpoint_callback(
                output,
                {
                    "epoch": checkpoint_epoch,
                    "step": step,
                    "status": "budget_stopped" if reason == "max_seconds" else "checkpoint_saved",
                    "stop_reason": reason,
                    "selected_by": metadata["selected_by"],
                    "final_selected": final,
                    "last_checkpoint": str(output / "last.pt"),
                    "best_checkpoint": str(output / "best.pt")
                    if (settings["validation"] or final) and (output / "best.pt").exists()
                    else None,
                },
            )

    for epoch in range(initial_epoch, settings["epochs"]):
        first_batch = initial_batch if epoch == initial_epoch else 0
        next_batch = first_batch
        if budget_expired():
            stop_reason = "max_seconds"
            metadata.update(
                trained=step > 0,
                completed_epoch=epoch,
                total_optimizer_steps=step,
                complete_last_epoch=first_batch == 0,
                resume_cursor={"epoch_index": epoch, "next_batch_index": first_batch},
                stop_reason=stop_reason,
            )
            persist(checkpoint_epoch=epoch, reason=stop_reason)
            break
        train_generator.manual_seed(settings["seed"] + epoch)
        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device=device)
        batches_seen, accumulated, epoch_samples = (0, 0, 0)
        epoch_start = time.perf_counter()
        for batch_index, batch in enumerate(loader):
            if batch_index < first_batch:
                continue
            batch = _move_batch(batch, device, settings["non_blocking"])
            micro_step = epoch * len(loader) + batch_index
            loss, components = _loss(model, batch, settings, config["loss"], micro_step)
            _check_finite(loss, device, f"Non-finite loss at epoch {epoch + 1}, batch {batch_index}")
            (loss / settings["accumulation_steps"]).backward()
            accumulated += 1
            epoch_samples += batch["image"].shape[0]
            loss_sum.add_(loss.detach())
            batches_seen += 1
            next_batch = batch_index + 1
            if accumulated == settings["accumulation_steps"] or batch_index == len(loader) - 1:
                if accumulated != settings["accumulation_steps"]:
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(settings["accumulation_steps"] / accumulated)
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings["clip_grad_norm"])
                optimizer.step()
                if hasattr(model, "constrain_parameters"):
                    model.constrain_parameters()
                optimizer.zero_grad(set_to_none=True)
                accumulated = 0
                step += 1
                if settings["log_every"] and step % settings["log_every"] == 0:
                    logged_loss, logged_components = _log_components(loss, components)
                    print(
                        json.dumps(
                            {
                                "epoch": epoch + 1,
                                "step": step,
                                "loss": logged_loss,
                                "components": logged_components,
                            }
                        ),
                        flush=True,
                    )
                if settings["max_steps"] is not None and step >= settings["max_steps"]:
                    stop_reason = "max_steps"
                    break
                if budget_expired():
                    stop_reason = "max_seconds"
                    break
        synchronize(device)
        train_seconds = time.perf_counter() - epoch_start
        complete_epoch = next_batch == len(loader)
        val = None
        if settings["validation"] and stop_reason != "max_seconds":
            val = validation_loss(
                model,
                val_loader,
                device,
                settings["stage"],
                settings["seed"] + 900000,
                settings["non_blocking"],
            )
        row = {
            "epoch": epoch + 1,
            "optimizer_step": step,
            "train_loss": float((loss_sum / max(1, batches_seen)).cpu()),
            "validation_l1": val,
            "samples": epoch_samples,
            "train_seconds": train_seconds,
            "samples_per_second": epoch_samples / max(train_seconds, 1e-09),
            "complete_epoch": complete_epoch,
            "start_batch_index": first_batch,
            "next_batch_index": next_batch,
            "batches": batches_seen,
        }
        if device.type == "mps":
            row["mps_driver_bytes"] = torch.mps.driver_allocated_memory()
        history.append(row)
        total_samples += epoch_samples
        print(json.dumps(row), flush=True)
        improved = val is not None and val < best - 1e-07
        if val is not None:
            if improved:
                best, failures = (val, 0)
            else:
                failures += 1
        if stop_reason is None and complete_epoch and (epoch + 1 == settings["epochs"]):
            stop_reason = "epochs"
        if stop_reason is None and settings["validation"] and (failures >= settings["patience"]):
            stop_reason = "early_stopping"
        final_selected = not settings["validation"] and stop_reason in ("epochs", "max_steps")
        cursor = {
            "epoch_index": epoch + 1 if complete_epoch else epoch,
            "next_batch_index": 0 if complete_epoch else next_batch,
        }
        metadata.update(
            trained=step > 0,
            best_val_loss=best if math.isfinite(best) else None,
            completed_epoch=epoch + 1 if complete_epoch else epoch,
            total_optimizer_steps=step,
            best_epoch=epoch + 1 if improved else metadata.get("best_epoch"),
            complete_last_epoch=complete_epoch,
            early_stopping_failures=failures,
            resume_cursor=cursor,
            stop_reason=stop_reason,
            final_selected=final_selected,
        )
        if final_selected:
            metadata.update(best_epoch=epoch + 1, selected_step=step)
        persist(
            checkpoint_epoch=epoch + 1 if complete_epoch else epoch,
            improved=improved,
            final=final_selected,
            reason=stop_reason,
        )
        if stop_reason is not None:
            break
    synchronize(device)
    summary = {
        **metadata,
        "status": "budget_stopped" if stop_reason == "max_seconds" else "completed",
        "best_checkpoint": str((output / "best.pt").resolve())
        if (settings["validation"] or final_selected) and (output / "best.pt").exists()
        else None,
        "last_checkpoint": str((output / "last.pt").resolve()),
        "final_checkpoint": str((output / "final.pt").resolve()) if final_selected else None,
        "elapsed_seconds": time.perf_counter() - started,
        "samples_processed_this_run": total_samples,
        "epochs_logged": len(history),
        "stopped_by_step_budget": stop_reason == "max_steps",
        "stopped_by_time_budget": stop_reason == "max_seconds",
        "convergence_claim": "Not established. Validation selection is disabled."
        if not settings["validation"]
        else "Not established; inspect held-out validation and matched baselines.",
    }
    atomic_json(output / "summary.json", summary)
    return summary


def benchmark(config: dict, steps: int = 200, output_dir="runs/benchmark", *, train_dataset=None) -> dict:
    """Measure microbatches using the configured gradient accumulation.

    Compare samples_per_second, not step latency, when batch sizes differ.
    Unlike train(), this synchronized diagnostic does not select checkpoints.
    """
    from .data import PairedDataset
    from .models import ModelConfig, build_model

    config = _resolve(config)
    if steps < 1:
        raise ValueError("steps must be positive")
    settings = config["training"]
    device = select_device(settings["device"])
    seed_everything(settings["seed"])
    torch.set_num_threads(settings["cpu_threads"] or min(4, torch.get_num_threads()))
    dataset = (
        train_dataset
        if train_dataset is not None
        else PairedDataset(
            config["data"]["manifest"],
            "train",
            image_size=config["data"]["image_size"],
            limit=config["data"].get("limit"),
        )
    )
    model_config = dict(config["model"])
    if "styles" not in model_config:
        names = {row["style_id"]: row["style_name"] for row in dataset.rows}
        if sorted(names) != list(range(len(names))):
            raise ValueError("Benchmark style IDs must be contiguous from zero")
        model_config["styles"] = tuple(names[index] for index in range(len(names)))
    if "styles" in model_config:
        model_config["styles"] = tuple(model_config["styles"])
    model = build_model(ModelConfig(**model_config)).to(device).train()
    loader = _loader(
        dataset, settings, device, shuffle=False, generator=torch.Generator().manual_seed(settings["seed"])
    )
    if not len(loader):
        raise ValueError("No training samples for benchmark")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    iterator = iter(loader)
    times, measured_samples, measured_updates = ([], 0, 0)
    accumulated, batch_index = (0, 0)
    optimizer.zero_grad(set_to_none=True)
    for index in range(steps + 5):
        synchronize(device)
        start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
            batch_index = 0
        batch = _move_batch(batch, device, settings["non_blocking"])
        loss, _ = _loss(model, batch, settings, config["loss"], index)
        _check_finite(loss, device, "Non-finite benchmark loss")
        (loss / settings["accumulation_steps"]).backward()
        accumulated += 1
        updated = (
            accumulated == settings["accumulation_steps"]
            or batch_index == len(loader) - 1
            or index == steps + 4
        )
        if updated:
            if accumulated != settings["accumulation_steps"]:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(settings["accumulation_steps"] / accumulated)
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["clip_grad_norm"])
            optimizer.step()
            if hasattr(model, "constrain_parameters"):
                model.constrain_parameters()
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0
        synchronize(device)
        if index >= 5:
            times.append(time.perf_counter() - start)
            measured_samples += batch["image"].shape[0]
            measured_updates += int(updated)
        batch_index += 1
        if (index + 1) % 25 == 0:
            print(json.dumps({"benchmark_steps": index + 1, "requested": steps}), flush=True)
    report = {
        "benchmark_version": "configured-microbatch-v2",
        "step_unit": "microbatch",
        "environment": environment_info(device),
        "config": config,
        "measured_steps": steps,
        "warmup_steps": 5,
        "mean_step_seconds": sum(times) / len(times),
        "median_step_seconds": sorted(times)[len(times) // 2],
        "batch_size": settings["batch_size"],
        "includes_decode_backward_optimizer": True,
        "effective_batch_size": settings["batch_size"] * settings["accumulation_steps"],
        "measured_samples": measured_samples,
        "measured_optimizer_updates": measured_updates,
        "samples_per_second": measured_samples / sum(times),
        "data_backend": type(dataset).__module__ + "." + type(dataset).__name__,
        "timing_note": "Synchronized microbatches, optimizer only at configured accumulation/epoch boundaries; Compare sample throughput across batch sizes. No validation or checkpoint timing included.",
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    if device.type == "mps":
        report["mps_driver_bytes"] = torch.mps.driver_allocated_memory()
    atomic_json(Path(output_dir) / "benchmark.json", report)
    return report
