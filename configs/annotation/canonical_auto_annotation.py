import os

source_root = os.environ.get(
    'FLUXMIMIR_ANNOTATION_SOURCE_ROOT',
    'work_dirs/datasets/aloha_smoke',
)
output_root = os.environ.get(
    'FLUXMIMIR_ANNOTATION_OUTPUT_ROOT',
    'work_dirs/datasets/aloha_smoke_annotated',
)

pipeline = dict(
    kind='annotation',
    runner=dict(
        type='AnnotationRunner',
        work_dir='work_dirs/annotation/canonical',
        resume=True,
        overwrite=False,
        idempotency_key='content+schema+prompt+provider+model',
    ),
    source=dict(
        type='CanonicalDatasetSource',
        root=source_root,
        expected_schema_id='agilex_aloha_unified',
        expected_schema_version='1.1.0',
        cameras=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        # Keep the default billable run to one episode. Override explicitly
        # after inspecting the smoke result and provider usage.
        max_episodes=1,
    ),
    stages=[
        dict(
            type='SampleAnnotationClips',
            clip_duration_s=3.0,
            frames_per_clip=4,
            max_clips=4,
            max_image_side=768,
            jpeg_quality=85,
            reject_static_clips=False,
        ),
        dict(
            type='AnnotateEpisodeSemantics',
            schema_id='fluxmimir.episode_semantics.v1',
            schema_version='1.0.0',
            prompt_version='episode_semantics_v1',
            output_language='English',
            provider=dict(
                type='OpenAICompatibleAnnotationProvider',
                api_key_env='ANNOTATION_API_KEY',
                base_url_env='ANNOTATION_BASE_URL',
                model_env='ANNOTATION_MODEL',
                api_mode='auto',
                timeout_s=120,
                max_retries=5,
                retry_backoff_s=1.0,
                max_tokens=4096,
                temperature=0.0,
                require_json_schema=True,
                record_usage=True,
            ),
        ),
        dict(
            type='RouteAmbiguousAnnotations',
            strategy='confidence_and_quality_review',
            confidence_below=0.7,
            auto_accept_above=0.9,
        ),
    ],
    writer=dict(
        type='CanonicalAnnotationReleaseWriter',
        root=output_root,
        update_mode='new_minor_release',
        in_place=False,
        hardlink_unchanged=True,
        store_raw_response_by_reference=False,
    ),
    validators=[
        dict(type='AnnotationJSONSchemaValidator'),
        dict(type='TaskSubtaskTemporalValidator'),
        dict(type='AnnotationProvenanceValidator'),
        dict(type='SecretLeakValidator'),
        dict(type='CanonicalParquetSchemaValidator'),
        dict(type='ForeignKeyClosureValidator'),
        dict(type='VideoAlignmentValidator'),
        dict(type='ArtifactManifestValidator'),
    ],
)
