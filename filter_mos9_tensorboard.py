"""Keep the current trainer running while relaying only the 16 dashboard tags."""
import argparse
from pathlib import Path
import signal
import time
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from algorithm.mos9_limit_action.tensorboard import SCALAR_TAGS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    loaders = {}
    writer = EventFileWriter(str(args.output), flush_secs=5)
    count = 0
    try:
        while not stopped:
            for path in sorted(args.source.glob('events.out.tfevents.*')):
                if path not in loaders:
                    loaders[path] = EventFileLoader(str(path))
                for event in loaders[path].Load():
                    values = [v for v in event.summary.value if v.tag in SCALAR_TAGS]
                    if values:
                        writer.add_event(Event(wall_time=event.wall_time, step=event.step,
                                               summary=Summary(value=values)))
                        count += len(values)
            writer.flush()
            time.sleep(2)
    finally:
        writer.close()
        print(f'Relayed {count} scalar events for {len(SCALAR_TAGS)} allowed tags', flush=True)


if __name__ == '__main__':
    main()
