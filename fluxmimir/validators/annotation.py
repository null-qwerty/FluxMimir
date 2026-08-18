from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from fluxmimir.annotation import (EPISODE_ANNOTATION_JSON_SCHEMA,
                                  validate_annotation_payload)
from fluxmimir.engines.utils import VALIDATORS
from fluxmimir.structures import (IssueSeverity, ValidationIssue,
                                  ValidationReport)

from .base import BaseValidator


def _annotation_paths(root: Path) -> list[Path]:
    return sorted((root / 'annotations/episodes').glob('episode_*.json'))


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


@VALIDATORS.register_module()
class AnnotationJSONSchemaValidator(BaseValidator):

    def validate(self, value: Any) -> ValidationReport:
        root = Path(value)
        issues = []
        paths = _annotation_paths(root)
        if not paths:
            issues.append(
                ValidationIssue('missing_annotations',
                                'annotation release has no records'))
        for path in paths:
            try:
                record = _load(path)
                validate_annotation_payload(record['payload'],
                                            EPISODE_ANNOTATION_JSON_SCHEMA)
            except Exception as exc:
                issues.append(
                    ValidationIssue(
                        'annotation_schema',
                        str(exc),
                        location=str(path),
                    ))
        return ValidationReport(
            issues=tuple(issues),
            metrics={'annotation_records': float(len(paths))},
        )


@VALIDATORS.register_module()
class TaskSubtaskTemporalValidator(BaseValidator):

    def validate(self, value: Any) -> ValidationReport:
        root = Path(value)
        issues = []
        for path in _annotation_paths(root):
            record = _load(path)
            episode_index = int(record['episode_index'])
            episode_rows = [
                json.loads(line)
                for line in (root / 'meta/episodes.jsonl').read_text(
                    encoding='utf-8').splitlines() if line.strip()
            ]
            episode = next(
                item for item in episode_rows
                if int(item['episode_index']) == episode_index)
            length = int(episode['length'])
            occupied = set()
            for subtask in record['payload']['subtasks']:
                start = int(subtask['start_frame'])
                end = int(subtask['end_frame'])
                if start > end or start < 0 or end >= length:
                    issues.append(
                        ValidationIssue(
                            'subtask_range_repaired',
                            f'AI range [{start}, {end}] is outside episode',
                            severity=IssueSeverity.WARNING,
                            location=str(path),
                        ))
                    continue
                frames = set(range(start, end + 1))
                if occupied & frames:
                    issues.append(
                        ValidationIssue(
                            'subtask_overlap_repaired',
                            'AI subtask ranges overlap',
                            severity=IssueSeverity.WARNING,
                            location=str(path),
                        ))
                occupied.update(frames)
            if len(occupied) < length:
                issues.append(
                    ValidationIssue(
                        'subtask_gap_unknown',
                        f'{length - len(occupied)} frames use unknown subtask',
                        severity=IssueSeverity.WARNING,
                        location=str(path),
                    ))
        return ValidationReport(issues=tuple(issues))


@VALIDATORS.register_module()
class AnnotationProvenanceValidator(BaseValidator):

    def validate(self, value: Any) -> ValidationReport:
        root = Path(value)
        issues = []
        required = ('request_id', 'episode_uid', 'schema_id', 'schema_version',
                    'provider', 'model', 'prompt_version', 'review_status',
                    'evidence', 'provenance')
        for path in _annotation_paths(root):
            record = _load(path)
            missing = [name for name in required if not record.get(name)]
            if missing:
                issues.append(
                    ValidationIssue(
                        'annotation_provenance',
                        f'missing provenance fields: {missing}',
                        location=str(path),
                    ))
            for evidence in record.get('evidence', []):
                if not evidence.get('image_sha256'):
                    issues.append(
                        ValidationIssue(
                            'annotation_evidence',
                            'evidence fingerprint is missing',
                            location=str(path),
                        ))
        return ValidationReport(issues=tuple(issues))


@VALIDATORS.register_module()
class SecretLeakValidator(BaseValidator):
    _PATTERNS = (
        re.compile(r'sk-[A-Za-z0-9._-]{16,}'),
        re.compile(r'(?i)authorization\s*[=:]\s*bearer\s+\S+'),
        re.compile(r'(?i)api[_-]?key\s*[=:]\s*["\']?\S{16,}'),
    )

    def validate(self, value: Any) -> ValidationReport:
        root = Path(value)
        issues = []
        for base in (root / 'meta', root / 'annotations'):
            if not base.exists():
                continue
            for path in base.rglob('*'):
                if not path.is_file() or path.stat().st_size > 5 * 1024 * 1024:
                    continue
                try:
                    text = path.read_text(encoding='utf-8')
                except UnicodeDecodeError:
                    continue
                if any(pattern.search(text) for pattern in self._PATTERNS):
                    issues.append(
                        ValidationIssue(
                            'secret_leak',
                            'possible credential material in release artifact',
                            location=str(path),
                        ))
        return ValidationReport(issues=tuple(issues))
