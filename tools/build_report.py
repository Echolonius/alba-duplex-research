"""Fills the report template from the measurement files.

  python -I build_report.py <localvoice_summary.json> <report_data.json> <out.html> "<verdict sentence>"
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))
import scale_math  # noqa: E402

lv_sum = json.loads(Path(sys.argv[1]).read_text(encoding='utf8'))
rep = json.loads(Path(sys.argv[2]).read_text(encoding='utf8'))
out_path, verdict = Path(sys.argv[3]), sys.argv[4]
d_sum = json.loads((ROOT / 'results' / 'dessa_summary.json').read_text(encoding='utf8'))
D, H, L = d_sum['test_with_people'], d_sum['people_with_test'], lv_sum['test_with_people']


def chip(kind):
    label = {'ahead': 'Ahead', 'close': 'Close', 'behind': 'Behind'}[kind]
    return f'<span class="chip {kind}">{label}</span>'


def judge(lv, ds, mode, tol):
    if mode == 'lower':
        return 'ahead' if lv < ds - tol else ('close' if lv <= ds + tol else 'behind')
    if mode == 'higher':
        return 'ahead' if lv > ds + tol else ('close' if lv >= ds - tol else 'behind')
    return 'close' if abs(lv - ds) <= tol else 'behind'


rows = [
    ('Starts its turn after the other stops (median)', lambda s: s['response_offset_s']['median'], '{:.2f} s', 'lower', 0.1),
    ('Turn starts that overlap the other\'s last words', lambda s: s['response_offset_s']['share_overlapping_start'], '{:.0%}', 'near', 0.15),
    ('Turn starts slower than 0.7 s', lambda s: s['response_offset_s']['share_over_700ms'], '{:.0%}', 'lower', 0.05),
    ('"Mm-hm"-type sounds per minute of the other\'s speech', lambda s: s['backchannels_per_min_of_other_speech'], '{:.1f}', 'near', 3.0),
    ('Interrupts per minute', lambda s: s['interruptions_per_min'], '{:.1f}', 'near', 0.5),
    ('Keeps talking when really interrupted (median)', lambda s: s['kept_talking_when_talked_over_s']['median'], '{:.2f} s', 'near', 0.15),
    ('Share of the talking', lambda s: s['talk_share'], '{:.0%}', 'near', 0.1),
    ('Audio detail (bandwidth)', lambda s: s['bandwidth_hz_median'], '{:,.0f} Hz', 'higher', 300),
    ('Pitch range (expressiveness)', lambda s: s['f0_range_semitones_median'], '{:.1f} semitones', 'near', 1.5),
]
html_rows = []
for name, f, fmt, mode, tol in rows:
    vals = []
    for s in (D, H, L):
        try:
            v = f(s)
            vals.append((v, fmt.format(v)))
        except (TypeError, KeyError):
            vals.append((None, '–'))
    k = judge(vals[2][0], vals[0][0], mode, tol) if vals[2][0] is not None and vals[0][0] is not None else 'close'
    html_rows.append(f'<tr><td>{name}</td><td class="n">{vals[0][1]}</td><td class="n">{vals[1][1]}</td>'
                     f'<td class="n">{vals[2][1]}</td><td>{chip(k)}</td></tr>')

math_rows = []
for name, n, hours in scale_math.CASES:
    c = scale_math.flops(n, hours)
    laptop = c / (scale_math.RTX3050_FP16 * scale_math.LAPTOP_UTIL)
    h100 = c / (scale_math.H100_DENSE_BF16 * scale_math.UTIL)
    lo, hi = (h100 / 3600 * p for p in scale_math.RENT_PER_H100_HOUR)
    cost = 'under $1' if hi < 1 else f'${lo:,.0f}–{hi:,.0f}'
    size = f'{n / 1e9:g}B' if n >= 1e9 else f'{n / 1e6:g}M'
    math_rows.append(f'<tr><td>{name} ({size} parameters)</td><td class="n">{hours:,} h</td>'
                     f'<td class="n">{scale_math.fmt_time(laptop)}</td><td class="n">{scale_math.fmt_time(h100)}</td>'
                     f'<td class="n">{cost}</td></tr>')

data = {'report': rep,
        'medians': {'Dessa': D['response_offset_s']['median'], 'People talking to Dessa': H['response_offset_s']['median'],
                    'LocalVoice': L['response_offset_s']['median']},
        'bw': {'Dessa': D['bandwidth_hz_median'], 'LocalVoice': L['bandwidth_hz_median']}}
html = (ROOT / 'tools' / 'report_template.html').read_text(encoding='utf8')
fills = {'/*DATA*/': json.dumps(data), '/*SCORE_ROWS*/': '\n'.join(html_rows), '/*MATH_ROWS*/': '\n'.join(math_rows),
         '/*DATE*/': '4 November 2026', '/*LV_CONV*/': str(L['conversations']), '/*LV_MIN*/': f"{L['minutes']:.1f}",
         '/*LV_N*/': str(L['response_offset_s']['n']), '/*D_RESP*/': f"{D['response_offset_s']['median']:.2f}",
         '/*VERDICT*/': verdict}
for key, val in fills.items():
    html = html.replace(key, val)
assert not any(k in html for k in fills), 'unfilled placeholder'
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(html, encoding='utf8')
print('wrote', out_path, len(html), 'bytes')
