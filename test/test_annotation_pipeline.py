import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from fluxmimir.annotation.provider import BaseAnnotationProvider
from fluxmimir.conversion import EpisodeRecord
from fluxmimir.engines.runners.annotation import AnnotationRunner
from fluxmimir.engines.utils import BACKENDS
from fluxmimir.structures import EpisodeRef, PipelineContext, StageResult
from fluxmimir.writers.canonical import CanonicalDatasetWriter


def complete_payload():
    return {
        'task':
        'Place the banana in the container',
        'task_category':
        'pick_and_place',
        'subtasks': [{
            'start_frame': 0,
            'end_frame': 3,
            'instruction': 'Move the banana into the container',
            'category': 'object_transfer',
            'required_effectors': ['left_arm'],
            'manipulated_objects': ['banana'],
            'goal_conditions': ['banana is inside the container'],
            'confidence': 0.95,
        }],
        'scene_description':
        'A banana and a container are on a table.',
        'task_progress_assessment': {
            'status': 'complete',
            'reason': 'The banana ends inside the container.',
            'confidence': 0.94,
        },
        'next_atomic_action':
        'Release the banana.',
        'objects': [{
            'name': 'banana',
            'initial_state': 'on the table',
            'final_state': 'inside the container',
            'state_changes': ['moved from table to container'],
            'confidence': 0.95,
        }],
        'outcome': {
            'label': 'success',
            'reason': 'The visible goal condition is satisfied.',
            'confidence': 0.93,
        },
        'quality_flags': [],
        'confidence':
        0.93,
        'abstained':
        False,
    }


class ArrayFrameReader:

    def __init__(self, frames):
        self.frames = frames
        self.shape = frames.shape[1:]

    def __len__(self):
        return len(self.frames)

    def __iter__(self):
        return iter(self.frames)


@BACKENDS.register_module(name='StaticTestAnnotationProvider', force=True)
class StaticTestAnnotationProvider(BaseAnnotationProvider):

    def setup(self, context):
        del context
        self.model = 'static-test-model'
        self.base_url = 'mock://annotation'

    def complete_json(self, *, request_id, messages, response_schema):
        del request_id, messages, response_schema
        return {
            'payload': complete_payload(),
            'provider': 'static_test',
            'model': self.model,
            'provider_request_id': 'static-request',
            'usage': {},
            'latency_s': 0.01,
            'retry_count': 0,
            'finish_reason': 'stop',
            'structured_mode': 'json_schema',
        }


def _canonical_source(root: Path):
    length = 4
    frames = np.zeros((length, 32, 32, 3), dtype=np.uint8)
    frames[1:, 8:24, 8:24, 0] = 200
    ref = EpisodeRef(
        episode_uid='episode-test-1',
        source_uri='/tmp/episode-test-1.hdf5',
        source_episode_index=0,
    )
    episode = EpisodeRecord(
        ref=ref,
        length=length,
        fields={},
        media={
            'cam_high': ArrayFrameReader(frames),
            'cam_left_wrist': ArrayFrameReader(frames),
            'cam_right_wrist': ArrayFrameReader(frames),
        },
        metadata={
            'relative_path': 'episode-test-1.hdf5',
            'schema_fingerprint': 'test-schema',
        },
    )
    zeros = np.zeros((length, 107), dtype=np.float32)
    masks = np.zeros((length, 107), dtype=np.bool_)
    value = {
        'observation.unified_107d': zeros,
        'observation.unified_107d_mask': masks,
        'observation.ego2cam': np.zeros((length, 3, 4, 4), dtype=np.float32),
        'observation.ego2cam_valid': np.zeros((length, 3), dtype=np.bool_),
        'target.unified_107d': zeros.copy(),
        'target.unified_107d_mask': masks.copy(),
        'target.source': np.full(length, 5, dtype=np.int8),
        'timestamp': np.arange(length, dtype=np.float32) / 30,
        'frame_index': np.arange(length, dtype=np.int64),
        'episode_index': np.zeros(length, dtype=np.int64),
        'index': np.arange(length, dtype=np.int64),
        'subtask_index': np.zeros(length, dtype=np.int64),
        '_video_encoding': {
            'enabled': True,
            'codec': 'h264',
            'fps': 30,
            'frame_count_must_match': True,
            'crf': 23,
            'gop_size': 30,
        },
    }
    writer = CanonicalDatasetWriter(
        root=root,
        dataset_id='annotation_test',
        dataset_version='0.1.0',
    )
    writer.setup(PipelineContext('convert-run', str(root / 'work')))
    writer.write_episode(episode, StageResult(value=value))
    writer.finalize({
        'source_root': '/tmp/source',
        'fps': 30,
        'action_semantics': 'unknown',
    })


def test_annotation_runner_creates_new_canonical_release(tmp_path):
    source = tmp_path / 'source'
    output = tmp_path / 'annotated'
    _canonical_source(source)
    runner = AnnotationRunner(
        source={
            'type': 'CanonicalDatasetSource',
            'root': str(source),
            'expected_schema_id': 'agilex_aloha_unified',
            'expected_schema_version': '1.1.0',
        },
        stages=[{
            'type': 'SampleAnnotationClips',
            'clip_duration_s': 1,
            'frames_per_clip': 2,
            'max_clips': 1,
        }, {
            'type': 'AnnotateEpisodeSemantics',
            'provider': {
                'type': 'StaticTestAnnotationProvider'
            },
        }, {
            'type': 'RouteAmbiguousAnnotations',
            'confidence_below': 0.7,
            'auto_accept_above': 0.9,
        }],
        writer={
            'type': 'CanonicalAnnotationReleaseWriter',
            'root': str(output),
        },
        validators=[
            {
                'type': 'AnnotationJSONSchemaValidator'
            },
            {
                'type': 'AnnotationProvenanceValidator'
            },
            {
                'type': 'SecretLeakValidator'
            },
            {
                'type': 'CanonicalParquetSchemaValidator'
            },
            {
                'type': 'ForeignKeyClosureValidator'
            },
            {
                'type': 'VideoAlignmentValidator'
            },
            {
                'type': 'ArtifactManifestValidator'
            },
        ],
        work_dir=tmp_path / 'annotation-work',
    )
    try:
        result = runner.run()
    finally:
        runner.close()

    assert result['episodes'] == 1
    info = json.loads((output / 'meta/info.json').read_text())
    assert info['dataset_version'] == '0.2.0'
    assert info['annotation_contract']['mode'] == 'ai_preannotation_proposal'
    record = json.loads((output / 'annotations/episodes/'
                         'episode_000000.json').read_text())
    assert record['payload']['task'] == 'Place the banana in the container'
    assert record['review_status'] == 'auto_accepted'
    table = pq.read_table(output / 'data/chunk-000/episode_000000.parquet')
    assert len(set(table['subtask_index'].to_pylist())) == 1
