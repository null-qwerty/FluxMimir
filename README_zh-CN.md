# FluxMimir

FluxMimir 是 Flux 系列中负责离线数据工作的 codebase：将机器人原始采集
数据转换为版本化 canonical 数据集，生成 ego-to-robot 视觉数据，并为
FluxVLA 提供可审计的自动标注。

> **当前状态：** 已实现 AgileX ALOHA HDF 到 canonical 107D Parquet 的可执行
> 转换，包括可选的流式 MP4 编码、metadata、mask-aware 统计、manifest 和落盘
> 校验；已实现 OpenAI-compatible 与阿里云 DashScope 原生多模态 API 自动预标注
> 和新版本落盘；视觉合成 backend 仍处于框架阶段。

## 三条能力线

1. 将 AgileX ALOHA HDF episode 转换为
   `agilex_aloha_unified@1.1.0` 的固定 107D Parquet/视频/metadata 契约。
2. 按 Qwen-RobotManip 思路搭建轨迹重定向、人体区域分割、时序修复、机械臂
   IK/渲染、深度估计与遮挡合成的可替换 backend 链。
3. 通过 provider-neutral API 接口生成 task、subtask 和具身推理标注，并保留
   schema 校验、provenance、review、幂等与成本信息。

FluxMimir 只负责离线数据处理；FluxVLA 继续负责模型侧数据读取、训练和推理，
FluxThemis 负责 rollout 与评测。

## 安装、预检与转换

```bash
conda create -n fluxmimir python=3.10 -y
conda activate fluxmimir
cd /root/projects/FluxMimir
pip install -e '.[conversion,vision,api,dev]'

fluxmimir-convert \
  configs/conversion/agilex_aloha_hdf_to_canonical_107d.py --dry-run
fluxmimir-synthesize \
  configs/synthesis/qwen_robotmanip_ego_to_robot.py --dry-run
fluxmimir-annotate \
  configs/annotation/canonical_auto_annotation.py --dry-run
fluxmimir-validate configs/validation/canonical_107d.py --dry-run
```

真实转换需要显式传入 `--execute`。用户提供的 raw 根目录包含约 2.9 万个大
HDF，建议先只转换一个 episode：

```bash
fluxmimir-convert \
  configs/conversion/agilex_aloha_hdf_to_canonical_107d.py \
  --execute \
  --cfg-options \
  pipeline.source.max_episodes=1 \
  pipeline.writer.root=work_dirs/datasets/aloha_smoke
```

只校验 HDF→Parquet 时可再加 `pipeline.stages.2.enabled=False`，跳过 RGB 视频
编码。确认冒烟结果和输出盘容量后，再移除 `pipeline.source.max_episodes`。
示例配置也支持通过 `FLUXMIMIR_*` 环境变量覆盖路径。

## AI 自动预标注

自动标注会从 canonical MP4 按时间采样，将同一时刻的三路相机拼成证据图，并用
一次结构化多模态请求生成：episode task、帧段 subtask、场景描述、任务进度、
下一原子动作、物体状态变化、结果建议和数据质量标签。输出是新的数据集版本，
不会修改输入 release，也不会让 AI 生成机器人数值状态或 action 语义。

### 1. 准备环境与输入数据

安装 conversion、vision 和 API 依赖：

```bash
conda activate fluxmimir
cd /root/projects/FluxMimir
pip install -e '.[conversion,vision,api]'
```

输入必须是已经转换完成的 canonical release，至少包含：

```text
<source_root>/
├── data/chunk-*/episode_*.parquet
├── videos/chunk-*/observation.images.*/episode_*.mp4
└── meta/
    ├── info.json
    ├── episodes.jsonl
    └── artifacts.sha256.json
```

可以直接使用前文转换得到的冒烟数据：

```bash
test -f work_dirs/datasets/aloha_smoke/meta/artifacts.sha256.json
```

### 2. 配置 API

密钥只能通过环境变量传入，不能写入配置、README、日志或数据集。为避免真实
密钥进入 shell history，可以使用隐藏输入。URL 末尾不要带中文逗号。使用阿里云
DashScope 原生 `/api/v1` 地址时：

```bash
read -rsp 'ANNOTATION_API_KEY: ' ANNOTATION_API_KEY
export ANNOTATION_API_KEY
echo
export ANNOTATION_BASE_URL='https://<你的实例>.cn-beijing.maas.aliyuncs.com/api/v1'
export ANNOTATION_MODEL='替换为已开通的多模态模型名'
```

使用 OpenAI-compatible 接口时，将 base URL 设置为 provider 提供的兼容地址，
例如：

```bash
export ANNOTATION_BASE_URL='https://example.com/compatible-mode/v1'
```

程序会自动区分 `/api/v1` 的 DashScope 原生协议与 OpenAI-compatible 协议。
DashScope 原生模式必须设置 `ANNOTATION_MODEL`。

### 3. 先执行 dry-run

dry-run 只检查配置，不读取视频、不上传图片，也不会产生 API 费用：

```bash
fluxmimir-annotate \
  configs/annotation/canonical_auto_annotation.py \
  --dry-run
```

### 4. 标注一个 episode

第一次运行务必限制为一个 episode，并明确指定输入和新的空输出目录：

```bash
fluxmimir-annotate \
  configs/annotation/canonical_auto_annotation.py \
  --execute \
  --cfg-options \
  pipeline.source.root=work_dirs/datasets/aloha_smoke \
  pipeline.source.max_episodes=1 \
  pipeline.writer.root=work_dirs/datasets/aloha_smoke_annotated
```

默认采样参数为：每个时间片段采 4 帧、最多 4 个片段、三路相机横向拼接。一次
episode 只发起一次综合标注请求。

### 5. 检查输出

成功后会生成一个新的 minor release，输入数据不会被修改：

```text
work_dirs/datasets/aloha_smoke_annotated/
├── annotations/
│   ├── index.jsonl
│   └── episodes/episode_000000.json
├── data/chunk-000/episode_000000.parquet
├── videos/chunk-000/observation.images.*/episode_000000.mp4
└── meta/
    ├── tasks.jsonl
    ├── subtasks.jsonl
    ├── episodes.jsonl
    ├── info.json
    ├── fluxvla_dataset_version.json
    └── artifacts.sha256.json
```

查看完整 AI proposal、task 和 subtask：

```bash
python -m json.tool \
  work_dirs/datasets/aloha_smoke_annotated/annotations/episodes/episode_000000.json

sed -n '1,5p' \
  work_dirs/datasets/aloha_smoke_annotated/meta/tasks.jsonl
sed -n '1,20p' \
  work_dirs/datasets/aloha_smoke_annotated/meta/subtasks.jsonl
```

每个 annotation record 包含 schema、provider/model、置信度、review 状态、证据
帧 hash、token usage 和请求 provenance。Parquet 中的 `subtask_index` 会同步更新，
但 AI 输出仍是 proposal，因此数据集保持 `training_ready=false`。

### 6. 调整采样和批量标注

长 episode 可以增加采样片段和每段帧数，但图片数量及费用也会增加：

```bash
fluxmimir-annotate \
  configs/annotation/canonical_auto_annotation.py \
  --execute \
  --cfg-options \
  pipeline.source.root=work_dirs/datasets/aloha_smoke \
  pipeline.source.max_episodes=10 \
  pipeline.stages.0.max_clips=6 \
  pipeline.stages.0.frames_per_clip=4 \
  pipeline.writer.root=work_dirs/datasets/aloha_batch_annotated
```

建议每次逐步增加 `max_episodes` 并检查 provider usage。输出目录必须为空且不能与
输入目录相同。已完成的请求会缓存在
`work_dirs/annotation/canonical/annotation_cache/`；相同 episode、图片、schema、
prompt、provider 和 model 再次运行时会复用缓存，不重复发起计费请求，但仍需使用
新的输出 release 目录。

### 7. Review 状态

- `auto_accepted`：所有置信度达到自动接受阈值。
- `pending`：结构合法，但建议人工抽查。
- `needs_review`：低置信度、严重质量标签或时间段修复。
- `abstained`：视觉证据不足，模型主动放弃。

subtask 时间段存在越界、重叠或空缺时不会静默丢弃：映射修复会写入 annotation
record，未覆盖帧使用显式 unknown subtask，并进入人工复核。

### 8. 常见问题

- `missing annotation credential`：未设置 `ANNOTATION_API_KEY`。
- `requires a multimodal model name`：未设置 `ANNOTATION_MODEL`。
- HTTP 404：确认 base URL 是 `/api/v1` 或 provider 的兼容 `/v1` 地址，不要重复
  添加 `chat/completions`。
- `annotation output root is not empty`：选择新的输出目录；已发布 release 不允许
  原地覆盖。
- `invalid JSON` 或 schema 校验失败：模型没有按
  `fluxmimir.episode_semantics.v1` 返回完整字段，检查模型是否支持视觉输入与 JSON
  输出。
- 大规模运行前应确认数据授权、图片上传范围、provider 数据留存策略和预算。

## 文档

- [架构与职责边界](docs/architecture.md)
- [Canonical 107D 数据契约](docs/data_contract.md)
- [HDF 转换规划与审计](docs/conversion.md)
- [Ego-to-robot 合成规划](docs/synthesis.md)
- [自动标注实现与契约](docs/annotation.md)
- [密钥与数据安全](docs/security.md)

未确认的语义不会被伪造成训练真值：raw action 默认输出为全零 target、全 false
mask 和 `source=unknown`；缺失的 eepose/ego2cam 同样以显式 mask 表达；缺失
timestamp 时按配置 FPS 合成并记录 provenance。因此当前输出可浏览、可审计，
但会保持 `training_ready=false`。

本项目使用 Apache License 2.0。
