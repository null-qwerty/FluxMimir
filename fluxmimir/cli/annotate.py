import json
import sys
from typing import Sequence

from fluxmimir.engines import build_runner_from_cfg

from .common import build_parser, describe_plan, load_config


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser(
        'annotation',
        'Pre-annotate canonical robot episodes with a multimodal API.',
    )
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Explicitly execute billable annotation API requests.',
    )
    args = parser.parse_args(argv)
    cfg = load_config(args, 'annotation')
    print(describe_plan(cfg, args.config))
    if args.dry_run:
        print('Dry run complete; no data was read or written.')
        print('No media was uploaded and no API request was issued.')
        return 0
    if not args.execute:
        print(
            'Annotation execution requires explicit --execute confirmation.',
            file=sys.stderr,
        )
        return 2
    pipeline = cfg.pipeline
    runner = build_runner_from_cfg(
        pipeline.runner,
        default_args={
            'source': pipeline.source,
            'stages': pipeline.stages,
            'writer': pipeline.writer,
            'validators': pipeline.get('validators', []),
        },
    )
    try:
        result = runner.run()
    finally:
        runner.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
