# FluxMimir

FluxMimir is the offline data codebase in the Flux ecosystem. It is designed to
turn raw robot recordings into versioned canonical datasets, derive
ego-to-robot visual data, and produce auditable automated annotations for
FluxVLA.

> **Current status:** AgileX ALOHA HDF to canonical 107D Parquet conversion is
> executable, including optional streaming MP4 encoding, metadata, masked
> statistics, manifests, and release validation. OpenAI-compatible and
> DashScope-native multimodal API pre-annotation and versioned release writing
> are executable. Synthesis backends remain scaffolded.

[中文说明](README_zh-CN.md)

## Scope

- Convert raw AgileX ALOHA HDF episodes into the fixed
  `agilex_aloha_unified@1.1.0` 107D Parquet/video/metadata contract.
- Build the Qwen-RobotManip-style ego-to-robot pipeline through replaceable
  retargeting, segmentation, inpainting, kinematics/rendering, depth, and
  compositing backends.
- Call provider-neutral multimodal APIs for schema-validated task, subtask, and
  embodied-reasoning annotation with provenance, review, idempotency, and cost
  controls.
- Validate masks, time alignment, metadata foreign keys, statistics, quality
  gates, and release manifests before FluxVLA consumes a dataset.

FluxMimir owns offline data processing only. FluxVLA owns model-side dataset
reading/training/inference, while FluxThemis owns rollout and evaluation.

## Installation

### Prerequisites

- Linux
- Git
- Miniconda or Anaconda
- Enough free space for the environment and converted datasets

Python 3.10 is the recommended and tested version. From a terminal, create an
isolated environment and install FluxMimir in editable mode:

```bash
cd /root/projects/FluxMimir

conda create --name fluxmimir python=3.10 pip --yes
conda activate fluxmimir

python -m pip install -e '.[conversion,vision,api,dev]'
```

The extras install the following capability groups:

- `conversion`: HDF5, canonical Parquet, image decoding, and video encoding.
- `vision`: image and mask processing dependencies.
- `api`: the provider-neutral HTTP client used by annotation integrations.
- `dev`: tests, formatting, linting, spell checking, and pre-commit hooks.

For a runtime-only HDF-to-Parquet environment, install just the conversion
extra:

```bash
python -m pip install -e '.[conversion]'
```

If `conda activate` is unavailable in a fresh shell, initialize Conda once and
restart the shell:

```bash
conda init
```

### Verify the installation

Run these commands from the repository root after activating the environment:

```bash
python --version
python -c "import fluxmimir; print(fluxmimir.__version__)"
fluxmimir-convert \
  configs/conversion/agilex_aloha_hdf_to_canonical_107d.py --dry-run
pytest -q
```

The tested environment reports Python 3.10, imports FluxMimir successfully,
loads the conversion plan, and passes the complete test suite.

## Quick start

Inspect the supplied plans without touching data or resolving credentials:

```bash
fluxmimir-convert \
  configs/conversion/agilex_aloha_hdf_to_canonical_107d.py --dry-run
fluxmimir-synthesize \
  configs/synthesis/qwen_robotmanip_ego_to_robot.py --dry-run
fluxmimir-annotate \
  configs/annotation/canonical_auto_annotation.py --dry-run
fluxmimir-validate configs/validation/canonical_107d.py --dry-run
```

Conversion requires an explicit `--execute` guard. Start with one episode (the
configured raw root contains tens of thousands of very large HDF files):

```bash
fluxmimir-convert \
  configs/conversion/agilex_aloha_hdf_to_canonical_107d.py \
  --execute \
  --cfg-options \
  pipeline.source.max_episodes=1 \
  pipeline.writer.root=work_dirs/datasets/aloha_smoke
```

Set `pipeline.stages.2.enabled=False` to validate only HDF-to-Parquet without
encoding RGB videos. Remove `pipeline.source.max_episodes` only after checking
the smoke output and selecting an output volume with sufficient capacity.
Synthesis and standalone validation commands still provide only their dry-run
plans.

Annotation requires an explicit `--execute` guard and defaults to one episode
to bound billable requests:

```bash
export ANNOTATION_API_KEY='replace-with-a-rotated-key'
export ANNOTATION_BASE_URL='https://example.com/api/v1'
export ANNOTATION_MODEL='replace-with-a-multimodal-model-name'

fluxmimir-annotate \
  configs/annotation/canonical_auto_annotation.py \
  --execute \
  --cfg-options \
  pipeline.source.max_episodes=1 \
  pipeline.writer.root=work_dirs/datasets/aloha_smoke_annotated
```

The annotation runner samples synchronized camera evidence and returns task,
temporal subtasks, scene and progress descriptions, next action, object state
changes, outcome proposals, and quality flags under a strict JSON Schema. It
creates a new canonical dataset version, updates task/subtask foreign keys, and
stores provenance, usage, confidence, evidence hashes, and review state. It
does not generate numeric robot state or action semantics.

## Automated annotation tutorial

Install the API, video, and canonical-data dependencies:

```bash
conda activate fluxmimir
cd /root/projects/FluxMimir
pip install -e '.[conversion,vision,api]'
```

Configure credentials only through environment variables. A base URL ending in
`/api/v1` selects DashScope-native multimodal requests; other provider URLs use
OpenAI-compatible chat completions unless `api_mode` is overridden.

```bash
read -rsp 'ANNOTATION_API_KEY: ' ANNOTATION_API_KEY
export ANNOTATION_API_KEY
echo
export ANNOTATION_BASE_URL='https://<workspace>.cn-beijing.maas.aliyuncs.com/api/v1'
export ANNOTATION_MODEL='replace-with-an-enabled-multimodal-model'
```

Inspect the plan without reading or uploading media:

```bash
fluxmimir-annotate \
  configs/annotation/canonical_auto_annotation.py \
  --dry-run
```

Run one billable smoke episode against an existing canonical release:

```bash
fluxmimir-annotate \
  configs/annotation/canonical_auto_annotation.py \
  --execute \
  --cfg-options \
  pipeline.source.root=work_dirs/datasets/aloha_smoke \
  pipeline.source.max_episodes=1 \
  pipeline.writer.root=work_dirs/datasets/aloha_smoke_annotated
```

Inspect the complete proposal and canonical dictionaries:

```bash
python -m json.tool \
  work_dirs/datasets/aloha_smoke_annotated/annotations/episodes/episode_000000.json
sed -n '1,5p' work_dirs/datasets/aloha_smoke_annotated/meta/tasks.jsonl
sed -n '1,20p' work_dirs/datasets/aloha_smoke_annotated/meta/subtasks.jsonl
```

To annotate a bounded batch or increase visual evidence, override the source
limit and sampler. One comprehensive API request is issued per episode:

```bash
fluxmimir-annotate \
  configs/annotation/canonical_auto_annotation.py \
  --execute \
  --cfg-options \
  pipeline.source.root=work_dirs/datasets/canonical_input \
  pipeline.source.max_episodes=10 \
  pipeline.stages.0.max_clips=6 \
  pipeline.stages.0.frames_per_clip=4 \
  pipeline.writer.root=work_dirs/datasets/annotated_batch
```

Output roots must be new and empty. Identical content/schema/prompt/provider/model
requests reuse `work_dirs/annotation/canonical/annotation_cache/`, but each
published dataset still requires a new release directory. Review states are
`auto_accepted`, `pending`, `needs_review`, and `abstained`. AI labels remain
auditable proposals, so annotation alone does not make a dataset
`training_ready=true`.

Paths are configurable through the `FLUXMIMIR_*` environment variables used in
the example configs. API configs contain only credential environment-variable
names. Never put a credential value in a config, command, log, or dataset.

## Repository map

```text
configs/       MMEngine pipeline wiring examples
docs/          architecture, data, synthesis, annotation, and security contracts
fluxmimir/     importable interfaces, registries, schemas, and CLI package
scripts/       thin Python entry points
test/          contract, registry, config, and CLI tests
datasets/      ignored local data mount/output placeholder
work_dirs/     ignored run artifacts
```

## Documentation

- [Architecture and ownership](docs/architecture.md)
- [Canonical 107D data contract](docs/data_contract.md)
- [HDF conversion plan and audit](docs/conversion.md)
- [Ego-to-robot synthesis plan](docs/synthesis.md)
- [Automated annotation implementation and contract](docs/annotation.md)
- [Secret and data security](docs/security.md)

## Development

```bash
pytest
flake8 fluxmimir test scripts
isort --check-only fluxmimir test scripts
yapf --diff --recursive fluxmimir test scripts
```

Unconfirmed semantics remain explicit instead of blocking browsable output:
raw action stays a zero target with a false mask and source `unknown`, missing
eepose/ego2cam stays masked, and absent timestamps use configured FPS with
provenance. Such output is intentionally marked `training_ready=false`.

## License

Apache License 2.0. See [LICENSE](LICENSE).
