import json

import httpx

from fluxmimir.annotation import (ECOT_JSON_SCHEMA,
                                  EPISODE_ANNOTATION_JSON_SCHEMA,
                                  TASK_SUBTASK_JSON_SCHEMA, AnnotationRequest,
                                  validate_annotation_payload)
from fluxmimir.backends.annotation_api import \
    OpenAICompatibleAnnotationProvider
from fluxmimir.structures import PipelineContext


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


def test_annotation_request_requires_idempotent_identity():
    request = AnnotationRequest(
        request_id='content-schema-prompt-model-hash',
        episode_uid='episode-1',
        schema_id='fluxmimir.task_subtask.v1',
        media_uris=('artifact://episode-1/cam-high', ),
    )

    assert request.privileged_context == {}


def test_built_in_annotation_schemas_are_structured():
    assert TASK_SUBTASK_JSON_SCHEMA['required'] == ['task', 'subtasks']
    assert ECOT_JSON_SCHEMA['required'] == [
        'scene_description',
        'task_progress_assessment',
        'next_atomic_action',
    ]
    assert 'objects' in EPISODE_ANNOTATION_JSON_SCHEMA['required']
    assert 'quality_flags' in EPISODE_ANNOTATION_JSON_SCHEMA['required']
    validate_annotation_payload(complete_payload())


def test_openai_compatible_provider_returns_structured_json(monkeypatch):
    payload = complete_payload()

    def handler(request):
        assert request.url.path == '/compatible-mode/v1/chat/completions'
        assert request.headers['authorization'] == 'Bearer test-secret'
        body = json.loads(request.content)
        assert body['response_format']['type'] == 'json_schema'
        return httpx.Response(
            200,
            headers={'x-request-id': 'provider-request-1'},
            json={
                'id':
                'completion-1',
                'model':
                'test-vl-model',
                'choices': [{
                    'message': {
                        'content': json.dumps(payload)
                    },
                    'finish_reason': 'stop',
                }],
                'usage': {
                    'prompt_tokens': 10,
                    'completion_tokens': 20,
                },
            },
        )

    monkeypatch.setenv('ANNOTATION_API_KEY', 'test-secret')
    monkeypatch.setenv('ANNOTATION_BASE_URL',
                       'https://annotation.example/compatible-mode/v1，')
    monkeypatch.setenv('ANNOTATION_MODEL', 'test-vl-model')
    provider = OpenAICompatibleAnnotationProvider(
        max_retries=0,
        transport=httpx.MockTransport(handler),
    )
    provider.setup(PipelineContext('run', '/tmp'))
    result = provider.complete_json(
        request_id='request-1',
        messages=[{
            'role': 'user',
            'content': 'Return JSON'
        }],
        response_schema=EPISODE_ANNOTATION_JSON_SCHEMA,
    )
    provider.close()

    assert result['payload'] == payload
    assert result['provider_request_id'] == 'provider-request-1'
    assert result['usage']['completion_tokens'] == 20


def test_dashscope_native_provider_converts_multimodal_messages(monkeypatch):
    payload = complete_payload()

    def handler(request):
        assert request.url.path == (
            '/api/v1/services/aigc/multimodal-generation/generation')
        body = json.loads(request.content)
        assert body['model'] == 'qwen-vl-test'
        content = body['input']['messages'][-1]['content']
        assert any('image' in item for item in content)
        assert any('JSON Schema' in item.get('text', '')
                   for message in body['input']['messages']
                   for item in message['content'])
        return httpx.Response(
            200,
            json={
                'request_id': 'dashscope-request-1',
                'output': {
                    'choices': [{
                        'message': {
                            'content': [{
                                'text': json.dumps(payload)
                            }]
                        },
                        'finish_reason': 'stop',
                    }]
                },
                'usage': {
                    'input_tokens': 10,
                    'output_tokens': 20,
                },
            },
        )

    monkeypatch.setenv('ANNOTATION_API_KEY', 'test-secret')
    monkeypatch.setenv('ANNOTATION_BASE_URL',
                       'https://annotation.example/api/v1，')
    monkeypatch.setenv('ANNOTATION_MODEL', 'qwen-vl-test')
    provider = OpenAICompatibleAnnotationProvider(
        max_retries=0,
        transport=httpx.MockTransport(handler),
    )
    provider.setup(PipelineContext('run', '/tmp'))
    result = provider.complete_json(
        request_id='request-1',
        messages=[{
            'role': 'system',
            'content': 'Return grounded JSON.',
        }, {
            'role':
            'user',
            'content': [{
                'type': 'text',
                'text': 'Evidence',
            }, {
                'type': 'image_url',
                'image_url': {
                    'url': 'data:image/jpeg;base64,AA=='
                },
            }],
        }],
        response_schema=EPISODE_ANNOTATION_JSON_SCHEMA,
    )
    provider.close()

    assert result['payload'] == payload
    assert result['provider'] == 'dashscope_native'
    assert result['provider_request_id'] == 'dashscope-request-1'
