"""Backfill and follow teacher collection charts without touching the trainer."""
import argparse
import json
from pathlib import Path
import signal
import time

from algorithm.mos9_limit_action.tensorboard import CollectionSummaryWriter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    config = json.loads((run/'config.json').read_text())
    target = config['baseline_training_budgets']['initial_row_target']
    horizon = config['environment']['max_steps']
    target = ((target+horizon-1)//horizon)*horizon
    output = run/'tensorboard'/'collection'
    if output.exists() and any(output.glob('events.out.tfevents.*')):
        parser.error('collection event files already exist; do not launch a duplicate relay')
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    writer = CollectionSummaryWriter(str(output), row_limit=target, flush_secs=5)
    try:
        with (run/'teacher_collection.jsonl').open() as source:
            while not stopped:
                while True:
                    position = source.tell()
                    line = source.readline()
                    if not line.endswith('\n'):
                        source.seek(position)
                        break
                    writer.add_episode(json.loads(line))
                writer.flush()
                (run/'collection_tensorboard_relay.json').write_text(json.dumps({
                    'episodes':writer.episodes, 'successful_rows':writer.accepted_rows,
                    'target_rows':target, 'output':str(output),
                    'complete':writer.accepted_rows >= target},indent=2)+'\n')
                if writer.accepted_rows >= target or (run/'status.json').exists():
                    break
                time.sleep(2)
    finally:
        writer.close()
    print(f'Collection charts: {writer.episodes} episodes, {writer.accepted_rows} successful rows',flush=True)


if __name__ == '__main__':
    main()
