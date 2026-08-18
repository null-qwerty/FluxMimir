from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from fluxmimir.structures import PipelineContext, StageResult

from ..utils import (RUNNERS, build_source_from_cfg, build_stage_from_cfg,
                     build_validator_from_cfg, build_writer_from_cfg)
from .base import BaseRunner


@RUNNERS.register_module()
class AnnotationRunner(BaseRunner):
    """Execute canonical source -> multimodal labels -> annotated release."""

    def __init__(
        self,
        source: Mapping[str, Any] | Any,
        stages: Sequence[Mapping[str, Any] | Any],
        writer: Mapping[str, Any] | Any,
        validators: Sequence[Mapping[str, Any] | Any] = (),
        work_dir: str | Path = 'work_dirs/annotation/canonical',
        resume: bool = True,
        overwrite: bool = False,
        idempotency_key: str | None = None,
    ) -> None:
        if overwrite:
            raise ValueError('annotation overwrite is unsupported')
        self.work_dir = Path(work_dir).expanduser().resolve()
        self.resume = bool(resume)
        self.idempotency_key = idempotency_key
        self.source = (
            build_source_from_cfg(source)
            if isinstance(source, Mapping) else source)
        self.stages = [
            (build_stage_from_cfg(item) if isinstance(item, Mapping) else item)
            for item in stages
        ]
        if isinstance(writer, Mapping):
            self.writer = build_writer_from_cfg(
                writer, default_args={'source_root': self.source.root})
        else:
            self.writer = writer
        self.validators = [(build_validator_from_cfg(item) if isinstance(
            item, Mapping) else item) for item in validators]
        self._context: PipelineContext | None = None

    def run_setup(self) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._context = PipelineContext(
            run_id=uuid.uuid4().hex,
            work_dir=str(self.work_dir),
            metadata={
                'resume': self.resume,
                'source_root': str(self.source.root),
                'idempotency_key': self.idempotency_key,
                'annotator_version': '0.2.0',
            },
        )
        for stage in self.stages:
            stage.setup(self._context)
        self.writer.setup(self._context)

    def run(self) -> Mapping[str, Any]:
        if self._context is None:
            self.run_setup()
        assert self._context is not None
        references = list(self.source.discover())
        artifact_count = 0
        warning_count = 0
        cache_hits = 0.0
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(references, desc='Annotating episodes', unit='ep')
        except ImportError:  # pragma: no cover
            iterator = references
        for reference in iterator:
            episode = self.source.open_episode(reference)
            value: Any = episode
            warnings: list[str] = []
            provenance: dict[str, Any] = {}
            for stage in self.stages:
                result = stage.process(value, self._context)
                if not isinstance(result, StageResult):
                    raise TypeError(f'{type(stage).__name__}.process must '
                                    'return StageResult')
                value = result.value
                warnings.extend(result.warnings)
                provenance.update(result.provenance)
                cache_hits += result.metrics.get('annotation_cache_hit', 0.0)
            artifacts = self.writer.write_episode(
                episode,
                StageResult(
                    value=value,
                    warnings=tuple(warnings),
                    provenance=provenance,
                ),
            )
            artifact_count += len(artifacts)
            warning_count += len(warnings)
        manifest = self.writer.finalize({
            **self._context.metadata,
            'run_id':
            self._context.run_id,
        })
        reports = [
            validator.validate(self.writer.root)
            for validator in self.validators
        ]
        errors = [
            issue for report in reports for issue in report.issues
            if issue.severity.value == 'error'
        ]
        if errors:
            summary = '; '.join(f'{issue.code}: {issue.message}'
                                for issue in errors[:10])
            raise RuntimeError(f'annotation validation failed: {summary}')
        return {
            'run_id': self._context.run_id,
            'source_root': str(self.source.root),
            'output_root': str(self.writer.root),
            'episodes': len(references),
            'artifacts': artifact_count + 1,
            'warnings': warning_count,
            'cache_hits': int(cache_hits),
            'manifest': manifest.uri,
        }

    def close(self) -> None:
        for stage in reversed(self.stages):
            stage.close()
        self.writer.close()
