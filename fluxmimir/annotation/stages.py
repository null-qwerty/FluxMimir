from __future__ import annotations

import base64
import hashlib
import io
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from fluxmimir.backends.annotation_api import \
    OpenAICompatibleAnnotationProvider  # noqa: F401
from fluxmimir.conversion import EpisodeRecord
from fluxmimir.engines.stages import BaseStage
from fluxmimir.engines.utils import STAGES, build_backend_from_cfg
from fluxmimir.structures import PipelineContext, StageResult

from .schemas import (EPISODE_ANNOTATION_JSON_SCHEMA,
                      EPISODE_ANNOTATION_SCHEMA_ID,
                      validate_annotation_payload)
from .structures import AnnotationRecord


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    temporary.replace(path)


@STAGES.register_module()
class SampleAnnotationClips(BaseStage):
    """Sample time evidence and combine all configured camera views."""

    def __init__(
        self,
        clip_duration_s: float | Sequence[float] = 3.0,
        frames_per_clip: int = 4,
        max_clips: int = 4,
        cameras: Sequence[str] | None = None,
        max_image_side: int = 768,
        jpeg_quality: int = 85,
        reject_static_clips: bool = False,
        static_threshold: float = 0.01,
    ) -> None:
        durations = ([clip_duration_s] if isinstance(
            clip_duration_s, (int, float)) else list(clip_duration_s))
        if not durations or min(float(item) for item in durations) <= 0:
            raise ValueError('clip_duration_s must be positive')
        if frames_per_clip < 2 or max_clips < 1:
            raise ValueError('invalid annotation sampling limits')
        self.clip_duration_s = max(float(item) for item in durations)
        self.frames_per_clip = int(frames_per_clip)
        self.max_clips = int(max_clips)
        self.cameras = tuple(cameras or ())
        self.max_image_side = int(max_image_side)
        self.jpeg_quality = int(jpeg_quality)
        self.reject_static_clips = bool(reject_static_clips)
        self.static_threshold = float(static_threshold)

    def _indices(self, length: int, fps: int) -> list[int]:
        clip_frames = max(1, int(round(self.clip_duration_s * fps)))
        starts = list(range(0, length, clip_frames))
        if len(starts) > self.max_clips:
            selected = np.linspace(
                0, len(starts) - 1, self.max_clips, dtype=np.int64)
            starts = [starts[int(index)] for index in selected]
        indices: list[int] = []
        for start in starts:
            end = min(length - 1, start + clip_frames - 1)
            sampled = np.linspace(
                start, end, self.frames_per_clip, dtype=np.int64)
            indices.extend(int(item) for item in sampled)
        return sorted(set(indices))

    def _contact_sheet(
        self,
        camera_frames: Sequence[tuple[str, np.ndarray]],
        frame_index: int,
    ) -> tuple[str, str, np.ndarray]:
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError as exc:  # pragma: no cover - vision extra
            raise RuntimeError('Annotation sampling requires Pillow') from exc
        panels = []
        font = ImageFont.load_default()
        for camera, array in camera_frames:
            image = Image.fromarray(np.asarray(array, dtype=np.uint8), 'RGB')
            scale = min(1.0, self.max_image_side / max(image.size))
            if scale < 1.0:
                image = image.resize(
                    (max(1, int(image.width * scale)),
                     max(1, int(image.height * scale))),
                    Image.Resampling.LANCZOS,
                )
            label_height = 28
            panel = Image.new('RGB',
                              (image.width, image.height + label_height),
                              'black')
            panel.paste(image, (0, label_height))
            draw = ImageDraw.Draw(panel)
            draw.text((8, 8),
                      f'{camera} | frame {frame_index}',
                      fill='white',
                      font=font)
            panels.append(panel)
        width = sum(panel.width for panel in panels)
        height = max(panel.height for panel in panels)
        sheet = Image.new('RGB', (width, height), 'black')
        offset = 0
        for panel in panels:
            sheet.paste(panel, (offset, 0))
            offset += panel.width
        buffer = io.BytesIO()
        sheet.save(
            buffer, format='JPEG', quality=self.jpeg_quality, optimize=True)
        encoded = buffer.getvalue()
        data_url = 'data:image/jpeg;base64,' + base64.b64encode(
            encoded).decode('ascii')
        preview = np.asarray(
            sheet.resize((64, 64)).convert('L'), dtype=np.float32)
        return data_url, hashlib.sha256(encoded).hexdigest(), preview

    def process(self, episode: Any, context: PipelineContext) -> StageResult:
        del context
        if not isinstance(episode, EpisodeRecord):
            raise TypeError('SampleAnnotationClips expects EpisodeRecord')
        if episode.length < 1:
            raise ValueError('cannot annotate an empty episode')
        fps = int(episode.metadata.get('fps', 30))
        indices = self._indices(episode.length, fps)
        cameras = self.cameras or tuple(episode.media)
        missing = sorted(set(cameras) - set(episode.media))
        if missing:
            raise ValueError(f'episode lacks requested cameras: {missing}')
        decoded = {
            camera: episode.media[camera].read_frames(indices)
            for camera in cameras
        }
        timestamps = np.asarray(episode.fields['timestamp']).reshape(-1)
        sampled_frames = []
        previews = []
        for frame_index in indices:
            camera_frames = [(camera, decoded[camera][frame_index])
                             for camera in cameras]
            data_url, digest, preview = self._contact_sheet(
                camera_frames, frame_index)
            sampled_frames.append({
                'frame_index': frame_index,
                'timestamp_s': float(timestamps[frame_index]),
                'cameras': list(cameras),
                'image_sha256': digest,
                'data_url': data_url,
            })
            previews.append(preview)
        motion_scores = []
        for previous, current in zip(previews, previews[1:]):
            motion_scores.append(
                float(np.mean(np.abs(current - previous)) / 255.0))
        static = bool(
            motion_scores and max(motion_scores) < self.static_threshold)
        warnings = ()
        if static:
            warnings = ('sampled visual evidence appears static', )
        if static and self.reject_static_clips:
            raise ValueError('sampled episode is static under configured rule')
        return StageResult(
            value={
                'episode': episode,
                'sampled_frames': sampled_frames,
                'sampling': {
                    'clip_duration_s': self.clip_duration_s,
                    'frames_per_clip': self.frames_per_clip,
                    'max_clips': self.max_clips,
                    'frame_indices': indices,
                    'cameras': list(cameras),
                    'static': static,
                    'motion_scores': motion_scores,
                },
            },
            warnings=warnings,
        )


@STAGES.register_module()
class AnnotateEpisodeSemantics(BaseStage):
    """Request all semantic pre-labels in one multimodal API call."""

    def __init__(
        self,
        provider: Mapping[str, Any],
        schema_id: str = EPISODE_ANNOTATION_SCHEMA_ID,
        schema_version: str = '1.0.0',
        prompt_version: str = 'episode_semantics_v1',
        output_language: str = 'English',
    ) -> None:
        self.provider = build_backend_from_cfg(provider)
        self.schema_id = schema_id
        self.schema_version = schema_version
        self.prompt_version = prompt_version
        self.output_language = output_language
        self._context: PipelineContext | None = None

    def setup(self, context: PipelineContext) -> None:
        self._context = context
        self.provider.setup(context)

    def _messages(self, value: Mapping[str, Any]) -> list[dict[str, Any]]:
        episode = value['episode']
        sampling = value['sampling']
        user_text = (
            'Annotate one robot-manipulation episode from sampled visual '
            'evidence. The images are sparse observations, not a continuous '
            'video. Do not invent robot joint values, action commands, camera '
            'calibration, or unseen events. Use inclusive frame '
            'boundaries.\n\n'
            f'Episode UID: {episode.ref.episode_uid}\n'
            f'Total frames: {episode.length}\n'
            f'FPS: {episode.metadata.get("fps", 30)}\n'
            f'Sampled frames: {sampling["frame_indices"]}\n'
            f'Output language for natural-language fields: '
            f'{self.output_language}.\n\n'
            'Return every field required by the JSON schema. Describe the '
            'episode-level task, temporal atomic subtasks, scene, progress, '
            'next atomic action, manipulated objects and state changes, '
            'success/failure/partial outcome proposal, and data-quality '
            'flags. '
            'Use abstained=true and uncertainty labels when evidence is '
            'insufficient. Confidence values must be between 0 and 1.')
        content: list[dict[str, Any]] = [{
            'type': 'text',
            'text': user_text,
        }]
        for sampled in value['sampled_frames']:
            content.append({
                'type':
                'text',
                'text': (f'Evidence frame {sampled["frame_index"]} at '
                         f'{sampled["timestamp_s"]:.3f}s'),
            })
            content.append({
                'type': 'image_url',
                'image_url': {
                    'url': sampled['data_url'],
                },
            })
        return [{
            'role':
            'system',
            'content':
            ('You are a conservative robotics data pre-annotation system. '
             'Ground every label in visible evidence and return only '
             'schema-compliant JSON.'),
        }, {
            'role': 'user',
            'content': content,
        }]

    def _request_id(self, value: Mapping[str, Any]) -> str:
        provider_identity = {
            'type': type(self.provider).__name__,
            'base_url': getattr(self.provider, 'base_url', None),
            'model': getattr(self.provider, 'model', None),
            'api_mode': getattr(self.provider, 'api_mode', None),
        }
        identity = {
            'episode_uid': value['episode'].ref.episode_uid,
            'evidence':
            [item['image_sha256'] for item in value['sampled_frames']],
            'schema_id': self.schema_id,
            'schema': EPISODE_ANNOTATION_JSON_SCHEMA,
            'prompt_version': self.prompt_version,
            'provider': provider_identity,
        }
        return _canonical_hash(identity)

    def process(self, value: Any, context: PipelineContext) -> StageResult:
        if not isinstance(value, Mapping) or 'episode' not in value:
            raise TypeError('AnnotateEpisodeSemantics expects sampled episode')
        request_id = self._request_id(value)
        cache_path = (
            Path(context.work_dir) / 'annotation_cache' / f'{request_id}.json')
        cache_hit = cache_path.is_file()
        if cache_hit:
            result = json.loads(cache_path.read_text(encoding='utf-8'))
        else:
            result = dict(
                self.provider.complete_json(
                    request_id=request_id,
                    messages=self._messages(value),
                    response_schema=EPISODE_ANNOTATION_JSON_SCHEMA,
                ))
            validate_annotation_payload(
                result.get('payload'), EPISODE_ANNOTATION_JSON_SCHEMA)
            _atomic_json(cache_path, result)
        payload = result.get('payload')
        validate_annotation_payload(payload, EPISODE_ANNOTATION_JSON_SCHEMA)
        evidence = tuple({
            'frame_index': item['frame_index'],
            'timestamp_s': item['timestamp_s'],
            'cameras': item['cameras'],
            'image_sha256': item['image_sha256'],
        } for item in value['sampled_frames'])
        record = AnnotationRecord(
            request_id=request_id,
            episode_uid=value['episode'].ref.episode_uid,
            schema_id=self.schema_id,
            schema_version=self.schema_version,
            payload=payload,
            provider=str(result.get('provider', 'unknown')),
            model=str(result.get('model', 'unknown')),
            prompt_version=self.prompt_version,
            confidence=float(payload['confidence']),
            evidence=evidence,
            usage=dict(result.get('usage', {})),
            latency_s=(float(result['latency_s'])
                       if result.get('latency_s') is not None else None),
            provenance={
                'provider_request_id': result.get('provider_request_id'),
                'finish_reason': result.get('finish_reason'),
                'retry_count': int(result.get('retry_count', 0)),
                'structured_mode': result.get('structured_mode'),
                'api_mode': result.get('api_mode'),
                'cache_hit': cache_hit,
                'sampling': dict(value['sampling']),
                'created_at_utc': datetime.now(timezone.utc).isoformat(),
            },
        )
        output = dict(value)
        output.pop('sampled_frames', None)
        output['annotation'] = asdict(record)
        return StageResult(
            value=output,
            metrics={'annotation_cache_hit': float(cache_hit)},
            provenance={
                'annotation_request_id': request_id,
                'annotation_model': record.model,
            },
        )

    def close(self) -> None:
        self.provider.close()


@STAGES.register_module()
class RouteAmbiguousAnnotations(BaseStage):

    def __init__(
        self,
        confidence_below: float = 0.7,
        auto_accept_above: float = 0.9,
        strategy: str = 'review',
    ) -> None:
        self.confidence_below = float(confidence_below)
        self.auto_accept_above = float(auto_accept_above)
        self.strategy = strategy

    @staticmethod
    def _confidences(payload: Mapping[str, Any]) -> list[float]:
        values = [float(payload['confidence'])]
        values.extend(
            float(item['confidence']) for item in payload['subtasks'])
        values.extend(float(item['confidence']) for item in payload['objects'])
        values.append(float(payload['task_progress_assessment']['confidence']))
        values.append(float(payload['outcome']['confidence']))
        return values

    def process(self, value: Any, context: PipelineContext) -> StageResult:
        del context
        if not isinstance(value, Mapping) or 'annotation' not in value:
            raise TypeError('RouteAmbiguousAnnotations expects annotation')
        annotation = dict(value['annotation'])
        payload = annotation['payload']
        confidences = self._confidences(payload)
        reasons = []
        if payload['abstained']:
            status = 'abstained'
            reasons.append('provider_abstained')
        elif any(float(item) < self.confidence_below for item in confidences):
            status = 'needs_review'
            reasons.append('confidence_below_threshold')
        elif any(item['severity'] == 'error'
                 for item in payload['quality_flags']):
            status = 'needs_review'
            reasons.append('error_quality_flag')
        elif min(confidences) >= self.auto_accept_above:
            status = 'auto_accepted'
        else:
            status = 'pending'
        annotation['review_status'] = status
        provenance = dict(annotation.get('provenance', {}))
        provenance.update({
            'routing_strategy': self.strategy,
            'review_reasons': reasons,
            'minimum_confidence': min(confidences),
        })
        annotation['provenance'] = provenance
        output = dict(value)
        output['annotation'] = annotation
        return StageResult(value=output)
