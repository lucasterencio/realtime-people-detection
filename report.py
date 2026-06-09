import json
import os
from datetime import datetime

import config


def _now_iso():
    return datetime.now().isoformat(timespec='seconds')


def load_report():
    if os.path.exists(config.REPORT_FILE):
        with open(config.REPORT_FILE, 'r') as f:
            return json.load(f)
    return {'sessions': [], 'totals': {}}


def save_report(report):
    with open(config.REPORT_FILE, 'w') as f:
        json.dump(report, f, indent=2)


def add_session(report, name, duration, start_ts, end_ts):
    session = {
        'name': name,
        'start': datetime.fromtimestamp(start_ts).isoformat(timespec='seconds'),
        'end': datetime.fromtimestamp(end_ts).isoformat(timespec='seconds'),
        'duration_seconds': round(duration, 1),
    }
    report['sessions'].append(session)
    if 'totals' not in report:
        report['totals'] = {}
    report['totals'][name] = round(report['totals'].get(name, 0) + duration, 1)
    save_report(report)
    return session


def get_totals(report):
    return report.get('totals', {})


def print_summary(report):
    totals = get_totals(report)
    if not totals:
        print("\nNenhum registro de permanencia.")
        return
    print("\n--- Relatorio de Permanencia (acumulado) ---")
    for name, total in sorted(totals.items()):
        print(f"  {name}: {total:.1f}s")
    print(f"  Total de sessoes: {len(report['sessions'])}")
    print(f"  Salvo em: {config.REPORT_FILE}")
