"""Bound sequential GPU command runtime against one shared dollar ledger.

This limits launched processes. Rental idle time/storage and provider billing
remain external; set the provider's job timeout as well.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path

from wmscale.common import load_config, write_json


def run(cfg, command, hourly_usd, label):
    if not math.isfinite(hourly_usd) or hourly_usd <= 0:
        raise ValueError('Supply the actual positive GPU hourly price')
    e = cfg['execution']
    ledger = Path(e.get('budget_ledger', 'scale_runs/budget.json'))
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger.with_suffix('.lock'), 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        data = json.loads(ledger.read_text()) if ledger.exists() else {'entries': []}
        used = sum(x.get('cost_usd', x['reserved_usd']) for x in data['entries'])
        cap = min(float(e.get('phase_cap_usd', 70)), float(e.get('total_cap_usd', 100)))
        remaining = cap - used
        seconds = min(int(e.get('max_command_seconds', 7200)), int(remaining * 3600 / hourly_usd))
        if seconds < 60:
            raise ValueError(f'Insufficient budget: ${used:.2f} accounted against ${cap:.2f} phase cap')
        entry = {'label': label, 'run_name': cfg['run_name'], 'hourly_usd': hourly_usd,
                 'reserved_usd': seconds * hourly_usd / 3600, 'started_unix': time.time(), 'status': 'running'}
        data['entries'].append(entry)
        write_json(ledger, data)
        print(f'[budget] ${used:.2f} used; command timeout {seconds}s at ${hourly_usd:.2f}/hour', flush=True)
        proc, started = None, time.monotonic()
        try:
            proc = subprocess.Popen(command, start_new_session=True)
            code = proc.wait(timeout=seconds)
            entry['status'] = 'complete' if code == 0 else 'failed'
        except subprocess.TimeoutExpired:
            entry['status'], code = 'budget_timeout', 124
        finally:
            if proc is not None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=10)
                except ProcessLookupError:
                    pass
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            entry['wall_seconds'] = time.monotonic() - started
            entry['cost_usd'] = entry['wall_seconds'] * hourly_usd / 3600
            if entry['status'] == 'running':
                entry['status'] = 'interrupted'
            write_json(ledger, data)
        return code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--hourly-usd', type=float, required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('command', nargs=argparse.REMAINDER)
    a = ap.parse_args()
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    if not command:
        ap.error('Supply a command after --')
    raise SystemExit(run(load_config(a.config), command, a.hourly_usd, a.label))


if __name__ == '__main__':
    main()
