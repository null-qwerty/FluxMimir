from __future__ import annotations

import json
import os
import time
from typing import Any, Mapping, Sequence

from fluxmimir.annotation.provider import BaseAnnotationProvider
from fluxmimir.engines.utils import BACKENDS
from fluxmimir.structures import PipelineContext


def _clean_json_text(value: str) -> str:
    text = value.strip()
    if text.startswith('```'):
        lines = text.splitlines()
        if lines and lines[0].startswith('```'):
            lines = lines[1:]
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        text = '\n'.join(lines).strip()
    return text


@BACKENDS.register_module()
class OpenAICompatibleAnnotationProvider(BaseAnnotationProvider):
    """OpenAI-compatible or DashScope-native multimodal JSON client.

    Secrets are resolved only from ``api_key_env``. They are never retained in
    returned metadata, config snapshots, cache keys, or exception messages.
    """

    def __init__(
        self,
        api_key_env: str = 'ANNOTATION_API_KEY',
        base_url_env: str = 'ANNOTATION_BASE_URL',
        model_env: str = 'ANNOTATION_MODEL',
        base_url: str | None = None,
        model: str | None = None,
        api_mode: str = 'auto',
        endpoint_path: str | None = None,
        timeout_s: float = 120,
        max_retries: int = 5,
        retry_backoff_s: float = 1.0,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        require_json_schema: bool = True,
        record_usage: bool = True,
        transport: Any = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError('max_retries must be non-negative')
        self.api_key_env = api_key_env
        self.base_url_env = base_url_env
        self.model_env = model_env
        self._configured_base_url = base_url
        self._configured_model = model
        if api_mode not in ('auto', 'openai', 'dashscope'):
            raise ValueError('api_mode must be auto, openai, or dashscope')
        self._configured_api_mode = api_mode
        self.endpoint_path = endpoint_path.strip(
            '/') if endpoint_path else None
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self.retry_backoff_s = float(retry_backoff_s)
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.require_json_schema = bool(require_json_schema)
        self.record_usage = bool(record_usage)
        self.transport = transport
        self.base_url: str | None = None
        self.model: str | None = None
        self.api_mode: str | None = None
        self._client: Any = None

    def setup(self, context: PipelineContext) -> None:
        del context
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - optional API extra
            raise RuntimeError('Annotation API calls require the api extra') \
                from exc
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f'missing annotation credential environment variable '
                f'{self.api_key_env!r}')
        raw_base_url = (
            self._configured_base_url or os.environ.get(self.base_url_env, ''))
        raw_base_url = raw_base_url.strip().rstrip('/,，')
        if not raw_base_url.startswith(('http://', 'https://')):
            raise RuntimeError(
                f'missing or invalid annotation API base URL; set '
                f'{self.base_url_env!r}')
        self.base_url = raw_base_url
        if self._configured_api_mode == 'auto':
            if ('/api/v1' in raw_base_url
                    and '/compatible-mode/' not in raw_base_url):
                self.api_mode = 'dashscope'
            else:
                self.api_mode = 'openai'
        else:
            self.api_mode = self._configured_api_mode
        configured_model = (self._configured_model
                            or os.environ.get(self.model_env, '')).strip()
        self.model = configured_model or 'endpoint-default'
        if self.api_mode == 'dashscope' and self.model == 'endpoint-default':
            raise RuntimeError(
                f'DashScope native API requires a multimodal model name; set '
                f'{self.model_env!r}')
        self._client = httpx.Client(
            timeout=self.timeout_s,
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json',
            },
            transport=self.transport,
        )

    @property
    def endpoint(self) -> str:
        if self.base_url is None:
            raise RuntimeError('provider.setup() was not called')
        endpoint_path = self.endpoint_path
        if endpoint_path is None:
            endpoint_path = ('services/aigc/multimodal-generation/generation'
                             if self.api_mode == 'dashscope' else
                             'chat/completions')
        if self.base_url.rstrip('/').endswith(endpoint_path):
            return self.base_url
        return f'{self.base_url.rstrip("/")}/{endpoint_path}'

    def capabilities(self) -> dict[str, Any]:
        return {
            'multimodal_images':
            True,
            'structured_json':
            True,
            'json_schema_requested': (self.require_json_schema
                                      and self.api_mode == 'openai'),
            'api_mode':
            self.api_mode,
            'batching':
            False,
        }

    def estimate_cost(self, request: Any) -> Mapping[str, Any]:
        del request
        return {
            'status': 'unknown_provider_pricing',
            'currency': None,
            'amount': None,
        }

    def _request_body(
        self,
        messages: Sequence[Mapping[str, Any]],
        response_schema: Mapping[str, Any],
        structured_mode: str,
    ) -> dict[str, Any]:
        schema_text = json.dumps(
            response_schema,
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
        )
        messages_with_schema = list(messages)
        if structured_mode != 'json_schema':
            messages_with_schema = self._append_schema_instruction(
                messages_with_schema, schema_text)
        if self.api_mode == 'dashscope':
            return {
                'model': self.model,
                'input': {
                    'messages': self._dashscope_messages(messages_with_schema),
                },
                'parameters': {
                    'result_format': 'message',
                    'temperature': self.temperature,
                    'max_tokens': self.max_tokens,
                },
            }
        body: dict[str, Any] = {
            'messages': messages_with_schema,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
        }
        if self.model != 'endpoint-default':
            body['model'] = self.model
        if structured_mode == 'json_schema':
            body['response_format'] = {
                'type': 'json_schema',
                'json_schema': {
                    'name': 'fluxmimir_episode_annotation',
                    'strict': True,
                    'schema': dict(response_schema),
                },
            }
        elif structured_mode == 'json_object':
            body['response_format'] = {'type': 'json_object'}
        return body

    @staticmethod
    def _append_schema_instruction(
        messages: Sequence[Mapping[str, Any]],
        schema_text: str,
    ) -> list[dict[str, Any]]:
        result = [dict(message) for message in messages]
        instruction = (
            '\nReturn only one JSON object matching this JSON Schema exactly: '
            f'{schema_text}')
        for message in result:
            if message.get('role') == 'system' \
                    and isinstance(message.get('content'), str):
                message['content'] += instruction
                break
        else:
            result.insert(0, {'role': 'system', 'content': instruction})
        return result

    @staticmethod
    def _dashscope_messages(messages: Sequence[Mapping[str, Any]]):
        output = []
        for message in messages:
            content = message.get('content')
            if isinstance(content, str):
                parts = [{'text': content}]
            else:
                parts = []
                for item in content or []:
                    if item.get('type') == 'text':
                        parts.append({'text': item.get('text', '')})
                    elif item.get('type') == 'image_url':
                        image_url = item.get('image_url', {})
                        url = image_url.get('url') if isinstance(
                            image_url, Mapping) else image_url
                        parts.append({'image': url})
            output.append({'role': message.get('role'), 'content': parts})
        return output

    @staticmethod
    def _extract_message(data: Mapping[str, Any]) -> tuple[Any, str | None]:
        choices = data.get('choices')
        if isinstance(choices, list) and choices:
            choice = choices[0]
            message = choice.get('message', {})
            return message.get('content'), choice.get('finish_reason')
        output = data.get('output')
        if isinstance(output, Mapping):
            output_choices = output.get('choices')
            if isinstance(output_choices, list) and output_choices:
                choice = output_choices[0]
                message = choice.get('message', {})
                return message.get('content'), choice.get('finish_reason')
            return output.get('text', output.get('content')), \
                output.get('finish_reason')
        if 'data' in data and isinstance(data['data'], Mapping):
            nested = data['data']
            return nested.get('content', nested.get('text')), \
                nested.get('finish_reason')
        raise RuntimeError('annotation API response has no message content')

    @staticmethod
    def _parse_content(content: Any) -> Mapping[str, Any]:
        if isinstance(content, Mapping):
            return dict(content)
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, Mapping):
                    if isinstance(part.get('json'), Mapping):
                        return dict(part['json'])
                    if isinstance(part.get('text'), str):
                        text_parts.append(part['text'])
            content = ''.join(text_parts)
        if not isinstance(content, str):
            raise RuntimeError('annotation API returned non-text JSON content')
        try:
            parsed = json.loads(_clean_json_text(content))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                'annotation API returned invalid JSON content') from exc
        if not isinstance(parsed, Mapping):
            raise RuntimeError('annotation API JSON must be an object')
        return dict(parsed)

    def complete_json(
        self,
        *,
        request_id: str,
        messages: Sequence[Mapping[str, Any]],
        response_schema: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if self._client is None:
            raise RuntimeError('provider.setup() was not called')
        if self.api_mode == 'dashscope':
            modes = ['dashscope_json']
        else:
            modes = (['json_schema', 'json_object']
                     if self.require_json_schema else ['json_object'])
        started = time.monotonic()
        last_error: Exception | None = None
        retries = 0
        response = None
        for mode_index, mode in enumerate(modes):
            body = self._request_body(messages, response_schema, mode)
            for attempt in range(self.max_retries + 1):
                try:
                    response = self._client.post(
                        self.endpoint,
                        json=body,
                        headers={'Idempotency-Key': request_id},
                    )
                    if (response.status_code == 400 and mode == 'json_schema'
                            and mode_index + 1 < len(modes)):
                        break
                    if response.status_code in (408, 409, 425, 429) \
                            or response.status_code >= 500:
                        if attempt < self.max_retries:
                            retries += 1
                            time.sleep(self.retry_backoff_s * (2**attempt))
                            continue
                    response.raise_for_status()
                    data = response.json()
                    content, finish_reason = self._extract_message(data)
                    payload = self._parse_content(content)
                    provider_request_id = (
                        response.headers.get('x-request-id')
                        or data.get('request_id') or data.get('id'))
                    return {
                        'payload':
                        payload,
                        'provider': ('dashscope_native' if self.api_mode
                                     == 'dashscope' else 'openai_compatible'),
                        'model':
                        data.get('model') or self.model,
                        'provider_request_id':
                        provider_request_id,
                        'usage':
                        data.get('usage', {}) if self.record_usage else {},
                        'latency_s':
                        time.monotonic() - started,
                        'retry_count':
                        retries,
                        'finish_reason':
                        finish_reason,
                        'structured_mode':
                        mode,
                        'api_mode':
                        self.api_mode,
                    }
                except Exception as exc:  # httpx types are optional at import
                    last_error = exc
                    status = getattr(response, 'status_code', None)
                    retryable = status in (408, 409, 425, 429) \
                        or (isinstance(status, int) and status >= 500) \
                        or status is None
                    if retryable and attempt < self.max_retries:
                        retries += 1
                        time.sleep(self.retry_backoff_s * (2**attempt))
                        continue
                    break
            if response is not None and response.status_code == 400 \
                    and mode == 'json_schema':
                continue
            break
        status = getattr(response, 'status_code', None)
        suffix = f' (HTTP {status})' if status is not None else ''
        raise RuntimeError(f'annotation API request failed{suffix}') \
            from last_error

    def annotate(self, request: Any) -> Mapping[str, Any]:
        raise NotImplementedError(
            'use complete_json for schema-bound requests')

    def annotate_batch(self, requests: Sequence[Any]) -> list[Any]:
        return [self.annotate(request) for request in requests]

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
