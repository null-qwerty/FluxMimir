from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from fluxmimir.annotation import EPISODE_ANNOTATION_JSON_SCHEMA
from fluxmimir.conversion import EpisodeRecord
from fluxmimir.engines.utils import WRITERS
from fluxmimir.structures import ArtifactRecord, PipelineContext, StageResult

from .base import BaseWriter


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    temporary.replace(path)


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(
        ''.join(
            json.dumps(value, ensure_ascii=False) + '\n' for value in values),
        encoding='utf-8',
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]


def _next_minor(version: str) -> str:
    parts = version.split('.')
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise ValueError(f'dataset version is not SemVer: {version!r}')
    major, minor, _ = (int(part) for part in parts)
    return f'{major}.{minor + 1}.0'


def _slug(value: str, prefix: str) -> str:
    digest = hashlib.sha256(value.encode('utf-8')).hexdigest()[:16]
    return f'{prefix}_{digest}'


@WRITERS.register_module()
class CanonicalAnnotationReleaseWriter(BaseWriter):
    """Create a new canonical release containing auditable AI proposals."""

    def __init__(
        self,
        root: str | Path,
        source_root: str | Path,
        update_mode: str = 'new_minor_release',
        dataset_version: str | None = None,
        in_place: bool = False,
        hardlink_unchanged: bool = True,
        store_raw_response_by_reference: bool = False,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.source_root = Path(source_root).expanduser().resolve()
        if in_place or self.root == self.source_root:
            raise ValueError('annotation must create a new release')
        self.update_mode = update_mode
        self.hardlink_unchanged = bool(hardlink_unchanged)
        self.store_raw_response_by_reference = bool(
            store_raw_response_by_reference)
        self.source_info = _read_json(self.source_root / 'meta/info.json')
        source_version = str(self.source_info['dataset_version'])
        if dataset_version is not None:
            self.dataset_version = dataset_version
        elif update_mode == 'new_minor_release':
            self.dataset_version = _next_minor(source_version)
        else:
            raise ValueError(f'unsupported update_mode: {update_mode}')
        self.context: PipelineContext | None = None
        self.episodes: list[dict[str, Any]] = []
        self.tasks: list[dict[str, Any]] = []
        self.subtasks: list[dict[str, Any]] = []
        self.annotations: list[dict[str, Any]] = []
        self._task_indices: dict[tuple[str, str], int] = {}
        self._subtask_indices: dict[tuple[str, str], int] = {}
        self._video_count = 0

    def setup(self, context: PipelineContext) -> None:
        self.context = context
        manifest = self.root / 'meta/artifacts.sha256.json'
        if manifest.exists():
            raise FileExistsError('refusing to mutate finalized annotation '
                                  f'release: {self.root}')
        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError('annotation output root is not empty: '
                                  f'{self.root}')
        self.root.mkdir(parents=True, exist_ok=True)

    def _copy_unchanged(self, source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            return
        if self.hardlink_unchanged:
            try:
                os.link(source, destination)
                return
            except OSError:
                pass
        shutil.copy2(source, destination)

    def _task_index(self, annotation: Mapping[str, Any]) -> int:
        payload = annotation['payload']
        key = (str(payload['task']).strip(),
               str(payload['task_category']).strip())
        if key in self._task_indices:
            return self._task_indices[key]
        index = len(self.tasks)
        self._task_indices[key] = index
        self.tasks.append({
            'task_index': index,
            'task_id': _slug('|'.join(key), 'task'),
            'task': key[0],
            'language': 'en',
            'category': key[1],
            'placeholder_status': 'ai_proposed',
            'annotation_provenance': {
                'request_id': annotation['request_id'],
                'provider': annotation['provider'],
                'model': annotation['model'],
                'schema_id': annotation['schema_id'],
                'schema_version': annotation['schema_version'],
                'prompt_version': annotation['prompt_version'],
                'confidence': annotation['confidence'],
                'review_status': annotation['review_status'],
            },
        })
        return index

    def _subtask_index(
        self,
        subtask: Mapping[str, Any],
        annotation: Mapping[str, Any],
    ) -> int:
        key = (str(subtask['instruction']).strip(),
               str(subtask['category']).strip())
        if key in self._subtask_indices:
            return self._subtask_indices[key]
        index = len(self.subtasks)
        self._subtask_indices[key] = index
        self.subtasks.append({
            'subtask_index':
            index,
            'subtask_id':
            _slug('|'.join(key), 'subtask'),
            'subtask':
            key[0],
            'language':
            'en',
            'category':
            key[1],
            'required_effectors':
            list(subtask['required_effectors']),
            'object_metadata': {
                'manipulated_objects': list(subtask['manipulated_objects']),
            },
            'goal_conditions':
            list(subtask['goal_conditions']),
            'placeholder_status':
            'ai_proposed',
            'annotation_provenance': {
                'request_id': annotation['request_id'],
                'provider': annotation['provider'],
                'model': annotation['model'],
                'confidence': subtask['confidence'],
                'review_status': annotation['review_status'],
            },
        })
        return index

    def _unknown_subtask(self, annotation: Mapping[str, Any]) -> int:
        return self._subtask_index(
            {
                'instruction': 'unknown atomic subtask',
                'category': 'unknown',
                'required_effectors': [],
                'manipulated_objects': [],
                'goal_conditions': [],
                'confidence': 0.0,
            }, annotation)

    def _update_review_status(self, request_id: str, status: str) -> None:
        for row in (*self.tasks, *self.subtasks):
            provenance = row.get('annotation_provenance', {})
            if provenance.get('request_id') == request_id:
                provenance['review_status'] = status

    def _frame_subtasks(
        self,
        length: int,
        annotation: dict[str, Any],
    ) -> tuple[list[int], list[dict[str, Any]]]:
        indices = [-1] * length
        occupied = [False] * length
        repairs: list[dict[str, Any]] = []
        subtasks = sorted(
            annotation['payload']['subtasks'],
            key=lambda item:
            (int(item['start_frame']), int(item['end_frame'])),
        )
        for order, subtask in enumerate(subtasks):
            original_start = int(subtask['start_frame'])
            original_end = int(subtask['end_frame'])
            start = max(0, original_start)
            end = min(length - 1, original_end)
            if start > end:
                repairs.append({
                    'type': 'invalid_range_skipped',
                    'subtask_order': order,
                    'range': [original_start, original_end],
                })
                continue
            subtask_index = self._subtask_index(subtask, annotation)
            overlap = False
            for frame_index in range(start, end + 1):
                if occupied[frame_index]:
                    overlap = True
                    continue
                indices[frame_index] = subtask_index
                occupied[frame_index] = True
            if (start, end) != (original_start, original_end):
                repairs.append({
                    'type': 'range_clamped',
                    'subtask_order': order,
                    'original': [original_start, original_end],
                    'used': [start, end],
                })
            if overlap:
                repairs.append({
                    'type': 'overlap_preserved_first_label',
                    'subtask_order': order,
                    'range': [start, end],
                })
        if not all(occupied):
            unknown = self._unknown_subtask(annotation)
            indices = [unknown if item < 0 else item for item in indices]
            repairs.append({
                'type': 'uncovered_frames_use_unknown',
                'count': occupied.count(False),
            })
        return indices, repairs

    def _write_parquet(
        self,
        source: Path,
        destination: Path,
        subtask_indices: list[int],
    ) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - conversion extra
            raise RuntimeError('Annotation writing requires pyarrow') from exc
        table = pq.read_table(source)
        if table.num_rows != len(subtask_indices):
            raise ValueError('annotation length does not match Parquet rows')
        column_index = table.column_names.index('subtask_index')
        table = table.set_column(
            column_index,
            'subtask_index',
            pa.array(subtask_indices, type=pa.int64()),
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + '.tmp')
        try:
            pq.write_table(table, temporary, compression='zstd')
            temporary.replace(destination)
        finally:
            if temporary.exists():
                temporary.unlink()

    def write_episode(self, episode: EpisodeRecord,
                      result: StageResult) -> tuple[ArtifactRecord, ...]:
        if self.context is None:
            raise RuntimeError('writer.setup() was not called')
        if not isinstance(result.value, Mapping):
            raise TypeError('annotation writer expects a mapping')
        annotation = dict(result.value['annotation'])
        episode_index = int(episode.metadata['episode_index'])
        task_index = self._task_index(annotation)
        subtask_indices, repairs = self._frame_subtasks(
            episode.length, annotation)
        if repairs:
            annotation['review_status'] = 'needs_review'
            provenance = dict(annotation.get('provenance', {}))
            provenance['temporal_repairs'] = repairs
            annotation['provenance'] = provenance
            self._update_review_status(annotation['request_id'],
                                       annotation['review_status'])
        source_data = Path(episode.metadata['data_path'])
        relative_data = source_data.relative_to(self.source_root)
        output_data = self.root / relative_data
        self._write_parquet(source_data, output_data, subtask_indices)
        artifacts = [
            ArtifactRecord('parquet', str(output_data), _sha256(output_data))
        ]
        for camera, source_uri in episode.metadata['video_paths'].items():
            source_video = Path(source_uri)
            output_video = self.root / source_video.relative_to(
                self.source_root)
            self._copy_unchanged(source_video, output_video)
            self._video_count += 1
            artifacts.append(
                ArtifactRecord(
                    'video',
                    str(output_video),
                    _sha256(output_video),
                    {'camera': camera},
                ))
        source_episode = dict(episode.metadata['source_episode'])
        outcome = annotation['payload']['outcome']
        source_episode.update({
            'task_index':
            task_index,
            'outcome':
            outcome['label'],
            'outcome_status':
            'ai_proposed',
            'outcome_reason':
            outcome['reason'],
            'outcome_confidence':
            outcome['confidence'],
            'annotation_request_id':
            annotation['request_id'],
            'annotation_review_status':
            annotation['review_status'],
        })
        self.episodes.append(source_episode)
        annotation['episode_index'] = episode_index
        annotation['temporal_repairs'] = repairs
        annotation['source_dataset'] = {
            'root': str(self.source_root),
            'dataset_id': episode.metadata.get('source_dataset_id'),
            'dataset_version': episode.metadata.get('source_dataset_version'),
        }
        annotation_path = (
            self.root / 'annotations/episodes' /
            f'episode_{episode_index:06d}.json')
        _write_json(annotation_path, annotation)
        self.annotations.append({
            'episode_index':
            episode_index,
            'episode_uid':
            episode.ref.episode_uid,
            'request_id':
            annotation['request_id'],
            'schema_id':
            annotation['schema_id'],
            'schema_version':
            annotation['schema_version'],
            'review_status':
            annotation['review_status'],
            'confidence':
            annotation['confidence'],
            'record_path':
            annotation_path.relative_to(self.root).as_posix(),
        })
        artifacts.append(
            ArtifactRecord('annotation', str(annotation_path),
                           _sha256(annotation_path)))
        return tuple(artifacts)

    def _filtered_episode_stats(self) -> list[dict[str, Any]]:
        selected = {int(item['episode_index']) for item in self.episodes}
        rows = _read_jsonl(self.source_root / 'meta/episodes_stats.jsonl')
        return [row for row in rows if int(row['episode_index']) in selected]

    def _build_info(self) -> dict[str, Any]:
        info = dict(self.source_info)
        blockers = [
            item for item in info.get('training_readiness_blockers', [])
            if item != 'episode-level task and frame subtasks are unavailable'
        ]
        blockers.append(
            'AI-generated semantic labels are proposals pending review')
        info.update({
            'dataset_version':
            self.dataset_version,
            'training_ready':
            False,
            'training_readiness_blockers':
            blockers,
            'total_episodes':
            len(self.episodes),
            'total_frames':
            sum(int(item['length']) for item in self.episodes),
            'total_tasks':
            len(self.tasks),
            'total_subtasks':
            len(self.subtasks),
            'total_videos':
            self._video_count,
            'total_chunks':
            len({
                int(item['episode_index']) //
                int(self.source_info.get('chunks_size', 1000))
                for item in self.episodes
            }),
            'splits': {
                'train': f'0:{len(self.episodes)}'
            },
            'annotation_contract': {
                'mode':
                'ai_preannotation_proposal',
                'schema_id':
                'fluxmimir.episode_semantics.v1',
                'schema_version':
                '1.0.0',
                'records': ('annotations/episodes/'
                            'episode_{episode_index:06d}.json'),
                'index':
                'annotations/index.jsonl',
                'review_states': [
                    'pending', 'auto_accepted', 'needs_review', 'rejected',
                    'abstained'
                ],
                'numeric_robot_state_generated_by_ai':
                False,
            },
            'parent_dataset': {
                'root':
                str(self.source_root),
                'dataset_id':
                self.source_info.get('dataset_id'),
                'dataset_version':
                self.source_info.get('dataset_version'),
                'manifest_sha256':
                _sha256(self.source_root / 'meta/artifacts.sha256.json'),
            },
        })
        return info

    def finalize(self, metadata: Mapping[str, Any]) -> ArtifactRecord:
        del metadata
        self.episodes.sort(key=lambda item: int(item['episode_index']))
        self.annotations.sort(key=lambda item: int(item['episode_index']))
        _write_jsonl(self.root / 'meta/episodes.jsonl', self.episodes)
        _write_jsonl(self.root / 'meta/episodes_stats.jsonl',
                     self._filtered_episode_stats())
        _write_jsonl(self.root / 'meta/tasks.jsonl', self.tasks)
        _write_jsonl(self.root / 'meta/subtasks.jsonl', self.subtasks)
        _write_jsonl(self.root / 'annotations/index.jsonl', self.annotations)
        info = self._build_info()
        _write_json(self.root / 'meta/info.json', info)
        now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        version = {
            'fluxvla_dataset_version': self.dataset_version,
            'dataset_id': info['dataset_id'],
            'qualified_dataset_id':
            f'{info["dataset_id"]}@{self.dataset_version}',
            'schema_id': info['schema_id'],
            'schema_version': info['schema_version'],
            'versioning_scheme': 'Semantic Versioning 2.0.0',
            'release_status': 'ai_preannotated_pending_review',
            'released_at_utc': now,
            'immutable_release': True,
            'content_manifest': 'meta/artifacts.sha256.json',
            'parent_dataset_version': self.source_info['dataset_version'],
        }
        _write_json(self.root / 'meta/fluxvla_dataset_version.json', version)
        manifest_path = self.root / 'meta/artifacts.sha256.json'
        artifacts = {}
        for directory in ('data', 'videos', 'meta', 'annotations'):
            base = self.root / directory
            if not base.exists():
                continue
            for path in sorted(base.rglob('*')):
                if path.is_file() and path != manifest_path:
                    relative = path.relative_to(self.root).as_posix()
                    artifacts[relative] = {
                        'sha256': _sha256(path),
                        'bytes': path.stat().st_size,
                    }
        source_manifest = _read_json(self.source_root /
                                     'meta/artifacts.sha256.json')
        manifest = {
            'manifest_format': 'fluxvla-dataset-manifest-v1',
            'manifest_version': '1.0.0',
            'synthetic': False,
            'schema': {
                'id': info['schema_id'],
                'version': info['schema_version'],
                'rules_sha256': source_manifest['schema']['rules_sha256'],
            },
            'dataset': {
                'id': info['dataset_id'],
                'version': self.dataset_version,
                'content_sha256': _canonical_sha256(artifacts),
            },
            'producer': {
                'name': 'fluxmimir-annotation',
                'version': '0.2.0',
                'sha256': _canonical_sha256(EPISODE_ANNOTATION_JSON_SCHEMA),
            },
            'artifact_count': len(artifacts),
            'artifacts': artifacts,
        }
        _write_json(manifest_path, manifest)
        return ArtifactRecord('manifest', str(manifest_path),
                              _sha256(manifest_path))
