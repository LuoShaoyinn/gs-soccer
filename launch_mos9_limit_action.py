"""Detach one MOS9 experiment on GPU 0; forward arguments to the trainer."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, default=Path('runs/mos9_limit_action/run1'))
    args, extra = p.parse_known_args()
    root = Path(__file__).resolve().parent
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        p.error(f'run directory already exists: {run_dir}')
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    log_path = run_dir.parent / (run_dir.name+'.log')
    pid_path = run_dir.parent / (run_dir.name+'.pid')
    if log_path.exists() or pid_path.exists():
        p.error('log or PID file already exists; choose a new run directory')
    environment = {**os.environ, 'HIP_VISIBLE_DEVICES':'0', 'PYTHONUNBUFFERED':'1'}
    with log_path.open('x') as log:
        child = subprocess.Popen([sys.executable,str(root/'train_mos9_limit_action.py'),'--run-dir',str(run_dir),*extra],cwd=root,env=environment,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    pid_path.write_text(str(child.pid)+'\n')
    print(f'pid={child.pid}\nlog={log_path}\nrun={run_dir}')


if __name__ == '__main__':
    main()
