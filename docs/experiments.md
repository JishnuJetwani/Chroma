# Running and interpreting experiments

All supplied training configurations explicitly select Apple MPS. No cloud jobs or model downloads are dispatched by the matrix utility. Commands below run from the repository root with the prepared virtual environment.

## Protocol and local budget

The planned `*_pilot.json` defaults use one Expert C style, a 256-pixel analysis preview, 512-pixel training canvas, microbatch two, accumulation four, learning rate 0.0002, and a maximum of 200 optimizer steps. They limit training/validation data to 128/32 entries and stop after three unimproved validation epochs; repair runs for at most five epochs and the other stages for ten. These are pipeline and feasibility pilots; they are not fair full-benchmark model rankings. Do not choose a preferred paper result from their small test subsets.

`matrix_preview.json` is the replay protocol for the actual initial measurements: immutable `data/fivek/manifest.preview.jsonl` with 128 training, 32 validation and 32 test images; 256-pixel analysis **and rendering**; seed 42; patience eight; fixed/global ten epochs, repair five, and both joint variants ten. Resolved optimizer and loss defaults are included in `configs/preview/`. Its outputs go to `runs/preview_reproduction/` to preserve existing measured pilots. It replays the protocol, not bit-exact MPS results. Because data manifests are ignored by Git, use the tracked [preview split registry](../configs/splits/fivek_preview.json) and [three-editor registry](../configs/splits/fivek_multistyle.json) with the [rebuild/check commands](data.md#s24-loader-behavior-and-commands) before a fresh-clone replay. These preserve exact scene selections and original hashes as evidence; validation rejects source, split, style or encoded-asset changes and refuses to overwrite live manifests. Native A/B conversion is a replay of the same scene/style protocol, not a promise of byte-identical files across machines: ICC/profile timestamps or encoder metadata can change hashes despite equal decoded pixels. Both matrices resolve the working directory from their own location.

1. Run operator tests and overfit a tiny paired subset. Stop if gradients, identity rendering, valid masks, crop alignment or color encoding fail. Fix the pipeline before adding architecture complexity.
2. Run one fixed/global/local pilot each, sequentially. Measure seconds/update and memory before extrapolating. On the 24 GB machine, reduce microbatch before reducing the common training resolution; preserve effective batch via accumulation. Stop an experiment on nonfinite loss, memory exhaustion, persistent swapping, or no validation improvement under its configured patience. Never claim the M4 has 24 GB exclusively available to model tensors.
3. Extend only the best two viable models to a common full-training budget. Start with at most 5,000 updates per model and patience eight; create separate configs/output directories, set `data.limit` and `data.val_limit` to null, and record the actual limit reached. Use synchronized pilot throughput to choose a session budget before execution. No unattended sweep is necessary.
4. Only after real validation gains, run three seeds (42, 43, 44) for the principal baseline/candidate comparison. Reserve one final test evaluation per selected checkpoint. S24 is a separate phase after its alignment and style registry are verified.

Repeat a check only after a relevant code/model/data change. The final system is whichever candidate meets quality and local resource gates, including the simpler baseline if additional stages do not help.

## Explicit command matrix

```sh
.venv/bin/python -m photograder.experiments matrix configs/matrix.json
.venv/bin/python -m photograder.experiments matrix configs/matrix.json --execute --max-runs 1
.venv/bin/python -m photograder train --config configs/global_pilot.json
.venv/bin/python -m photograder train --config configs/repair_pilot.json --initialize runs/global_pilot/best.pt
.venv/bin/python -m photograder train --config configs/local_pilot.json --initialize runs/repair_pilot/best.pt
.venv/bin/python -m photograder train --config configs/local_no_preservation_pilot.json --initialize runs/repair_pilot/best.pt

# Replay the actual 256-pixel preview curriculum into fresh output paths.
.venv/bin/python -m photograder.experiments matrix configs/matrix_preview.json
.venv/bin/python -m photograder.experiments matrix configs/matrix_preview.json --execute
```

Use the actual checkpoint filename produced by the trainer if it differs. `--initialize` transfers compatible tensors, records the source hash, and is distinct from `--resume`. Verify the loaded tensor count. For final architecture comparison, either train all candidates from scratch or document shared pretraining and charge its cost consistently; mixing pretrained candidates with scratch baselines silently is unfair. A repair-only checkpoint bypasses the global branch during training and is not a trained style grader by itself.

| Configuration | Question |
|---|---|
| `fixed_pilot.json` | Does image adaptation beat one learned color table? |
| `global_pilot.json` | Main global LUT baseline. |
| `local_pilot.json` | Does spatial correction improve target agreement and difficult regions? |
| `local_no_preservation_pilot.json` | Does the outside-region preservation penalty reduce drift? Identity/no-op samples remain enabled, isolating that penalty. |
| `local_bilinear_pilot.json` | Is image-guided map upsampling useful at boundaries? |
| `local_specialists_pilot.json` | Are sign/tonal restrictions useful? Maps are not calibrated defect labels. |
| `cascade_pilot.json` | Does independent sequential analysis justify additional compute? |
| `unet_pilot.json` | Low-resolution RGB-residual comparator; not a full-resolution U-Net reproduction. |
| `bilateral_pilot.json` | Internal local-affine grid comparator; not an HDRNet reproduction. |
| `repair_pilot.json` | Can the repair branch correct known interventions without restyling? |

The default matrix orders fixed → global → repair → local → local without the outside-preservation penalty. Fixed and global start from scratch; repair initializes from global; both joint variants independently initialize from the **same** repair checkpoint. Other architectures remain disabled with no forced checkpoint dependency. Dry-run construction does not require future checkpoints to exist; execution creates them in sequence. To examine an additional row, run its config explicitly or enable it after the pilot gates. Test `order: local_first` against the default `global_first` in a separately named config. Match measured wall time as a second comparison alongside matched updates, since the cascade spends more compute per step. Charge global and repair pretraining to both joint variants when reporting cumulative training cost.

## Quantitative evaluation

```sh
.venv/bin/python -m photograder evaluate --checkpoint runs/global_pilot/best.pt --manifest data/fivek/manifest.jsonl --split test --output runs/global_pilot/evaluation --save-images
```

Check CLI help for any optional `--limit` or image-size overrides. Every report must include the source manifest hash, model/checkpoint identity, style registry, scene split, rendering version and input evaluation size. Reports contain per-image CSV and aggregate JSON/Markdown. Perfect matches have a 120 dB PSNR ceiling. CIE76 must not be labeled CIEDE2000. SSIM v2 uses Gaussian windows, population covariance and reflect padding, scoring only windows whose entire support is valid. This excludes centers within five pixels of invalid privacy/padding at window size 11 while retaining real image edges via reflected masks. Empty masks/no valid windows produce missing values, not success scores. Reports with the previous center-mask SSIM protocol are superseded and must not enter matched comparisons.

Python callers can run `evaluate(..., include_controls=True)` for synthetic locality measurements. The default evaluation uses the `edge_aware` mask family (excluded from default training), including combined faults, applied to original paired inputs. Both corruption and statistics exclude invalid pixels. Compare same-style clean-graded/corrupt-graded outputs with a boundary exclusion band; paired targets measure agreement, and `corruption_penalty_mae` measures extra target error relative to the clean-input output. Record seed and loader batch size for replay. For repair-only evaluation, use the repair bypass consistently. Keep global paired, repair-only and counterfactual results in separately labeled tables.

Inference timing in `evaluate` is synchronized model-only batch timing, includes first-call warmup, and excludes decoding/metrics/encoding. It is not an end-to-end latency claim. For product timing, separately benchmark warm and cold single-image runs at 1, 12 and 24 megapixels, with decode, analysis, full-resolution render and encode measured separately. Check tiled and untiled equality plus 100% crops before calling output high-resolution verified.

After evaluating the selected checkpoints with identical settings, generate a matched comparison:

```sh
.venv/bin/python -m photograder.experiments compare runs/global_pilot/evaluation runs/local_pilot/evaluation runs/local_no_preservation_pilot/evaluation --output-dir runs/matched_comparison
```

The first report is the reference. The helper rejects mismatched manifest hashes, splits, image sizes, evaluation seeds, devices, metric definitions, control protocols, style registries or image identities. It writes candidate-minus-reference deltas with **paired** scene-bootstrap intervals. It does not establish equal training budgets, test-set independence, novelty or human preference. Run the primary comparison on validation during development; reserve final held-out reports for the selected models. The four-image external DeepLPF smoke uses a different format/resolution and intentionally cannot enter this matched table.

## Blind comparisons

Make a JSON array of records with `scene_id`, unique `image_id`, `input_path`, `a_path` (baseline output), and `b_path` (candidate output). Use exactly matching image content, resolution and encoding. Provide original JPEG/PNG files without model names visible in the panels.

```sh
.venv/bin/python -m photograder.experiments blind comparisons.json evaluation/blind --seed 42
.venv/bin/python -m photograder.experiments votes evaluation/blind/votes.csv evaluation/blind/private_key.csv --candidate candidate --output evaluation/blind/results.json
```

Share only `sheets/` and `votes.csv`. Keep `private_key.csv` and `private_protocol.json` away from raters. Each rater gets the blank template and sets a stable anonymous `rater_id`; concatenate completed rows for analysis. Accepted answers are A, B, tie, or blank. The analyzer rejects duplicate rater/comparison answers. Ties and unanswered questions are counted separately. Confidence intervals resample scenes after averaging within images; they do not model population-level rater uncertainty.

Use a preregistered primary outcome (overall preference) and treat local correction/artifact outcomes as secondary. Report effect sizes, intervals, number of independent scenes and all training seeds, including null results. A large number of votes on a handful of images does not create a large image sample.

## External baselines

The optional `run_external_deeplpf` function executes only a caller-supplied command in a clean external checkout matching a full 40-character audited git commit. It neither installs an environment nor downloads weights. Record the checkpoint hash, author-training split, color pipeline, resize, output gamut handling, and any scene overlap. An author checkpoint run on our photos is an external-checkpoint comparison; a reproduction additionally requires following and validating the original training/evaluation protocol. Do not transfer the author's reported GPU timing to this Mac.

## Budget and resume accounting

The first global/fixed pilots each allocate 160 optimizer updates. The local lineage allocates 160 global updates, 80 repair updates, and 160 joint updates. Validation can select an earlier checkpoint within each allocation; do not equate a checkpoint's own `step` field with the entire training budget. The two preservation variants start from the same selected repair checkpoint.

`configs/global_matched_pilot.json` continues the original global model for 240 additional updates, giving the same 400 allocated updates as the local curriculum. Invoke it with `--initialize runs/global_pilot/best.pt`. Equal updates still do not mean equal wall time or identical objectives; report both these distinctions and parameter counts.

`configs/multistyle_pilot.json` is a separate, balanced 24-scene C/A/B demonstration. Initialize it from `runs/local_pilot/best.pt`; named style transfer preserves the C embedding and initializes A/B independently. It must not enter the single-style architecture table, and its twelve test renditions contain only four independent scenes.

Resuming a checkpoint retains the same dataset hash, style registry, architecture, loss settings and corruption version. A step-budget stop mid-epoch is recorded and resumes at the next epoch, so such a continuation is not an exact replay of the unfinished epoch.
