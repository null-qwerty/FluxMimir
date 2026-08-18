from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from fluxmimir.conversion import EpisodeRecord
from fluxmimir.engines.utils import SOURCES
from fluxmimir.structures import EpisodeRef

from .base import BaseSource


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]


class MP4FrameReader:
    """Decode selected RGB frames from one canonical episode video."""

    def __init__(self, path: str | Path, length: int) -> None:
        self.path = Path(path)
        self.length = int(length)

    def __len__(self) -> int:
        return self.length

    def read_frames(self, indices: Sequence[int]) -> dict[int, np.ndarray]:
        wanted = sorted(set(int(index) for index in indices))
        if not wanted:
            return {}
        if wanted[0] < 0 or wanted[-1] >= self.length:
            raise IndexError('requested video frame is outside the episode')
        try:
            import av
        except ImportError as exc:  # pragma: no cover - optional media path
            raise RuntimeError('Canonical video reading requires PyAV') \
                from exc
        remaining = set(wanted)
        frames: dict[int, np.ndarray] = {}
        with av.open(str(self.path), mode='r') as container:
            for index, frame in enumerate(container.decode(video=0)):
                if index in remaining:
                    frames[index] = np.ascontiguousarray(
                        frame.to_ndarray(format='rgb24'))
                    remaining.remove(index)
                    if not remaining:
                        break
        if remaining:
            raise ValueError(
                f'{self.path}: video is missing frames {sorted(remaining)}')
        return frames

    def read_frame(self, index: int) -> np.ndarray:
        return self.read_frames([index])[int(index)]


@SOURCES.register_module()
class CanonicalDatasetSource(BaseSource):
    """Read an immutable FluxMimir canonical release for annotation."""

    def __init__(
        self,
        root: str | Path,
        expected_schema_id: str | None = None,
        expected_schema_version: str | None = None,
        cameras: Sequence[str] | None = None,
        max_episodes: int | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(
                f'canonical dataset root does not exist: {self.root}')
        self.info = _read_json(self.root / 'meta/info.json')
        if (expected_schema_id is not None
                and self.info.get('schema_id') != expected_schema_id):
            raise ValueError('canonical schema id does not match config')
        if (expected_schema_version is not None and
                self.info.get('schema_version') != expected_schema_version):
            raise ValueError('canonical schema version does not match config')
        if max_episodes is not None and max_episodes < 1:
            raise ValueError('max_episodes must be positive')
        self.max_episodes = max_episodes
        available = [
            key.removeprefix('observation.images.')
            for key, value in self.info.get('features', {}).items()
            if value.get('dtype') == 'video'
            and key.startswith('observation.images.')
        ]
        self.cameras = tuple(cameras or available)
        missing = sorted(set(self.cameras) - set(available))
        if missing:
            raise ValueError(f'canonical dataset lacks cameras: {missing}')
        self._episodes = sorted(
            _read_jsonl(self.root / 'meta/episodes.jsonl'),
            key=lambda item: int(item['episode_index']),
        )
        if self.max_episodes is not None:
            self._episodes = self._episodes[:self.max_episodes]

    def _data_path(self, episode_index: int) -> Path:
        chunk = episode_index // int(self.info.get('chunks_size', 1000))
        return (self.root / f'data/chunk-{chunk:03d}' /
                f'episode_{episode_index:06d}.parquet')

    def _video_path(self, episode_index: int, camera: str) -> Path:
        chunk = episode_index // int(self.info.get('chunks_size', 1000))
        return (self.root / f'videos/chunk-{chunk:03d}' /
                f'observation.images.{camera}' /
                f'episode_{episode_index:06d}.mp4')

    def discover(self) -> Iterator[EpisodeRef]:
        for row in self._episodes:
            episode_index = int(row['episode_index'])
            data_path = self._data_path(episode_index)
            if not data_path.is_file():
                raise FileNotFoundError(data_path)
            yield EpisodeRef(
                episode_uid=str(row['episode_uid']),
                source_uri=str(data_path),
                source_episode_index=episode_index,
                metadata={'episode': dict(row)},
            )

    def open_episode(self, episode: EpisodeRef) -> EpisodeRecord:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - conversion extra
            raise RuntimeError('Canonical source requires pyarrow') from exc
        data_path = Path(episode.source_uri)
        table = pq.read_table(
            data_path,
            columns=['frame_index', 'timestamp', 'subtask_index'],
        )
        length = int(table.num_rows)
        episode_index = int(episode.source_episode_index or 0)
        media = {}
        video_paths = {}
        for camera in self.cameras:
            path = self._video_path(episode_index, camera)
            if not path.is_file():
                raise FileNotFoundError(path)
            media[camera] = MP4FrameReader(path, length)
            video_paths[camera] = str(path)
        source_row = dict(episode.metadata.get('episode', {}))
        return EpisodeRecord(
            ref=episode,
            length=length,
            fields={
                'frame_index': table['frame_index'].to_numpy(),
                'timestamp': table['timestamp'].to_numpy(),
                'subtask_index': table['subtask_index'].to_numpy(),
            },
            media=media,
            metadata={
                'episode_index': episode_index,
                'source_root': str(self.root),
                'data_path': str(data_path),
                'video_paths': video_paths,
                'fps': int(self.info.get('fps', 30)),
                'source_episode': source_row,
                'source_dataset_id': self.info.get('dataset_id'),
                'source_dataset_version': self.info.get('dataset_version'),
                'schema_id': self.info.get('schema_id'),
                'schema_version': self.info.get('schema_version'),
            },
        )
