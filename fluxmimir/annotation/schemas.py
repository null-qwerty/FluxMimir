TASK_SUBTASK_SCHEMA_ID = 'fluxmimir.task_subtask.v1'
ECOT_SCHEMA_ID = 'fluxmimir.ecot.v1'
EPISODE_ANNOTATION_SCHEMA_ID = 'fluxmimir.episode_semantics.v1'

TASK_SUBTASK_JSON_SCHEMA = {
    'type': 'object',
    'required': ['task', 'subtasks'],
    'properties': {
        'task': {
            'type': 'string'
        },
        'subtasks': {
            'type': 'array',
            'items': {
                'type': 'object',
                'required': ['start_frame', 'end_frame', 'instruction'],
                'properties': {
                    'start_frame': {
                        'type': 'integer',
                        'minimum': 0
                    },
                    'end_frame': {
                        'type': 'integer',
                        'minimum': 0
                    },
                    'instruction': {
                        'type': 'string'
                    },
                },
            },
        },
    },
}

ECOT_JSON_SCHEMA = {
    'type':
    'object',
    'required': [
        'scene_description',
        'task_progress_assessment',
        'next_atomic_action',
    ],
    'properties': {
        'scene_description': {
            'type': 'string'
        },
        'task_progress_assessment': {
            'type': 'object',
            'required': ['status', 'reason'],
            'properties': {
                'status': {
                    'enum': ['complete', 'not_yet_complete']
                },
                'reason': {
                    'type': 'string'
                },
            },
        },
        'next_atomic_action': {
            'type': 'string'
        },
    },
}


def _confidence_schema():
    return {
        'type': 'number',
        'minimum': 0.0,
        'maximum': 1.0,
    }


EPISODE_ANNOTATION_JSON_SCHEMA = {
    '$schema':
    'https://json-schema.org/draft/2020-12/schema',
    'type':
    'object',
    'additionalProperties':
    False,
    'required': [
        'task',
        'task_category',
        'subtasks',
        'scene_description',
        'task_progress_assessment',
        'next_atomic_action',
        'objects',
        'outcome',
        'quality_flags',
        'confidence',
        'abstained',
    ],
    'properties': {
        'task': {
            'type': 'string',
            'minLength': 1,
        },
        'task_category': {
            'type': 'string',
            'minLength': 1,
        },
        'subtasks': {
            'type': 'array',
            'items': {
                'type':
                'object',
                'additionalProperties':
                False,
                'required': [
                    'start_frame',
                    'end_frame',
                    'instruction',
                    'category',
                    'required_effectors',
                    'manipulated_objects',
                    'goal_conditions',
                    'confidence',
                ],
                'properties': {
                    'start_frame': {
                        'type': 'integer',
                        'minimum': 0,
                    },
                    'end_frame': {
                        'type': 'integer',
                        'minimum': 0,
                    },
                    'instruction': {
                        'type': 'string',
                        'minLength': 1,
                    },
                    'category': {
                        'type': 'string',
                        'minLength': 1,
                    },
                    'required_effectors': {
                        'type': 'array',
                        'items': {
                            'type': 'string'
                        },
                    },
                    'manipulated_objects': {
                        'type': 'array',
                        'items': {
                            'type': 'string'
                        },
                    },
                    'goal_conditions': {
                        'type': 'array',
                        'items': {
                            'type': 'string'
                        },
                    },
                    'confidence': _confidence_schema(),
                },
            },
        },
        'scene_description': {
            'type': 'string',
            'minLength': 1,
        },
        'task_progress_assessment': {
            'type': 'object',
            'additionalProperties': False,
            'required': ['status', 'reason', 'confidence'],
            'properties': {
                'status': {
                    'enum': [
                        'not_started',
                        'in_progress',
                        'complete',
                        'failed',
                        'uncertain',
                    ],
                },
                'reason': {
                    'type': 'string',
                    'minLength': 1,
                },
                'confidence': _confidence_schema(),
            },
        },
        'next_atomic_action': {
            'type': 'string',
            'minLength': 1,
        },
        'objects': {
            'type': 'array',
            'items': {
                'type':
                'object',
                'additionalProperties':
                False,
                'required': [
                    'name',
                    'initial_state',
                    'final_state',
                    'state_changes',
                    'confidence',
                ],
                'properties': {
                    'name': {
                        'type': 'string',
                        'minLength': 1,
                    },
                    'initial_state': {
                        'type': 'string',
                        'minLength': 1,
                    },
                    'final_state': {
                        'type': 'string',
                        'minLength': 1,
                    },
                    'state_changes': {
                        'type': 'array',
                        'items': {
                            'type': 'string'
                        },
                    },
                    'confidence': _confidence_schema(),
                },
            },
        },
        'outcome': {
            'type': 'object',
            'additionalProperties': False,
            'required': ['label', 'reason', 'confidence'],
            'properties': {
                'label': {
                    'enum': ['success', 'failure', 'partial', 'uncertain'],
                },
                'reason': {
                    'type': 'string',
                    'minLength': 1,
                },
                'confidence': _confidence_schema(),
            },
        },
        'quality_flags': {
            'type': 'array',
            'items': {
                'type':
                'object',
                'additionalProperties':
                False,
                'required': [
                    'code',
                    'severity',
                    'description',
                    'start_frame',
                    'end_frame',
                ],
                'properties': {
                    'code': {
                        'type': 'string',
                        'minLength': 1,
                    },
                    'severity': {
                        'enum': ['info', 'warning', 'error'],
                    },
                    'description': {
                        'type': 'string',
                        'minLength': 1,
                    },
                    'start_frame': {
                        'type': ['integer', 'null'],
                        'minimum': 0,
                    },
                    'end_frame': {
                        'type': ['integer', 'null'],
                        'minimum': 0,
                    },
                },
            },
        },
        'confidence': _confidence_schema(),
        'abstained': {
            'type': 'boolean',
        },
    },
}


def validate_annotation_payload(payload, schema=None):
    """Validate provider JSON at the trust boundary."""
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover - optional API extra
        raise RuntimeError('Annotation validation requires `jsonschema`; '
                           'install the api extra') from exc
    jsonschema.validate(
        instance=payload,
        schema=schema or EPISODE_ANNOTATION_JSON_SCHEMA,
    )
