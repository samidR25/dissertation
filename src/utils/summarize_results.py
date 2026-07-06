"""
src/utils/summarize_results.py
================================
Scan results/*.json and print a clean markdown table.

Usage:
    python3 src/utils/summarize_results.py
    python3 src/utils/summarize_results.py --pattern "ann_results_multi_v2_seed*"
"""
import argparse, glob, json, os

parser = argparse.ArgumentParser()
parser.add_argument('--pattern', default='*results*.json')
parser.add_argument('--dir', default='results')
args = parser.parse_args()

paths = sorted(glob.glob(os.path.join(args.dir, args.pattern)))
if not paths:
    print(f"No files matched {args.dir}/{args.pattern}")
    raise SystemExit

rows = []
for p in paths:
    with open(p) as f:
        d = json.load(f)
    tag = os.path.basename(p).replace('.json', '')
    if 'event_level' in d:
        wl, el = d.get('window_level', {}), d.get('event_level', {})
        rows.append({'file': tag, 'split': 'window',
                     'n': None, 'n_seizure': None,
                     'sensitivity': wl.get('sensitivity'),
                     'specificity': wl.get('specificity'),
                     'fpr_per_hour': wl.get('fpr_per_hour')})
        rows.append({'file': tag, 'split': 'event',
                     'n': el.get('n_events'), 'n_seizure': el.get('n_detected'),
                     'sensitivity': el.get('event_sensitivity'),
                     'specificity': None,
                     'fpr_per_hour': el.get('fp_per_hour')})
    elif 'train' in d or 'val' in d or 'test' in d:
        for split in ('train', 'val', 'test'):
            s = d.get(split)
            if s:
                rows.append({'file': tag, 'split': split, **s})
    else:
        rows.append({
            'file': tag, 'split': d.get('eval_set', d.get('eval_patient', 'snn')),
            'n': d.get('n_eval'), 'n_seizure': d.get('n_seizure'),
            'sensitivity': d.get('sensitivity'), 'specificity': d.get('specificity'),
            'fpr_per_hour': d.get('fpr_per_hour'),
        })
headers = ['file', 'split', 'n', 'n_seizure', 'sensitivity', 'specificity', 'fpr_per_hour']

def fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    return '' if v is None else str(v)

print('| ' + ' | '.join(headers) + ' |')
print('|' + '|'.join(['---'] * len(headers)) + '|')
for r in rows:
    print('| ' + ' | '.join(fmt(r.get(h)) for h in headers) + ' |')
