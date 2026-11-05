"""Builds the README's figures (docs/assets/*.svg) from the saved results, each in a light and a dark version for
GitHub's two themes: the banner, hand-offs answered by build, and the audio work per 80 ms step before and after
the captured CUDA graphs. Standard library only.

  python tools/make_readme_figures.py
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / 'results'
OUT = ROOT / 'docs' / 'assets'
FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
# ink follows GitHub's own text colours; data marks: one accent blue and one neutral grey, checked for contrast and
# colour-blind separation on both page colours (#ffffff, #0d1117)
THEMES = {
    'light': {'ink': '#1f2328', 'ink2': '#59636e', 'grid': '#d8dee4', 'grey': '#898781', 'accent': '#2a78d6',
              'faint': '#c4c9cf'},
    'dark': {'ink': '#f0f6fc', 'ink2': '#9198a1', 'grid': '#30363d', 'grey': '#898781', 'accent': '#3987e5',
             'faint': '#3d444d'},
}


def esc(s):
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def text(x, y, s, size=14, fill='#000', weight=400, anchor='start'):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}">{esc(s)}</text>')


def hbar(x0, x1, y, h, fill, r=4):
    """A horizontal bar: square at the baseline, 4 px rounded at its data end."""
    r = min(r, (x1 - x0) / 2, h / 2)
    return (f'<path d="M{x0:.1f} {y:.1f}H{x1 - r:.1f}A{r} {r} 0 0 1 {x1:.1f} {y + r:.1f}V{y + h - r:.1f}'
            f'A{r} {r} 0 0 1 {x1 - r:.1f} {y + h:.1f}H{x0:.1f}Z" fill="{fill}"/>')


def svg(w, h, title, desc, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
            f'aria-labelledby="t d" font-family="{FONT}"><title id="t">{esc(title)}</title>'
            f'<desc id="d">{esc(desc)}</desc>{body}</svg>\n')


def banner(c):
    w, h = 1280, 300
    parts = [text(64, 128, 'Alba', 76, c['ink'], 700),
             text(64, 180, 'A full-duplex voice model on a 4 GB laptop GPU', 28, c['ink'], 500),
             text(64, 220, 'Listens and talks at the same time, in 80 ms steps, offline', 20, c['ink2'])]
    # two voices on one timeline: the caller above the line, Alba below; where both move, they overlap (full duplex)
    x0, step, n, mid, amp = 724, 9, 55, 150, 72
    parts.append(f'<line x1="{x0 - 8}" y1="{mid}" x2="{x0 + n * step}" y2="{mid}" stroke="{c["grid"]}" stroke-width="1"/>')
    for i in range(n):
        caller = (1.0 if i < 30 else 0.0) * (0.35 + 0.65 * abs(math.sin(i * 0.9) * math.cos(i * 0.37)))
        if 41 <= i <= 43:                                   # a short "mm-hm" while Alba talks
            caller = 0.22 + 0.08 * (i - 41)
        alba = (1.0 if 24 <= i < 54 else 0.0) * (0.35 + 0.65 * abs(math.sin(i * 0.73 + 1.1) * math.cos(i * 0.29)))
        x = x0 + i * step
        if caller > 0:
            hh = max(4.0, caller * amp)
            parts.append(f'<rect x="{x}" y="{mid - 2 - hh:.1f}" width="6" height="{hh:.1f}" rx="3" fill="{c["grey"]}"/>')
        if alba > 0:
            hh = max(4.0, alba * amp)
            parts.append(f'<rect x="{x}" y="{mid + 2:.1f}" width="6" height="{hh:.1f}" rx="3" fill="{c["accent"]}"/>')
    parts.append(text(x0 - 8, 62, 'caller', 18, c['ink2']))
    parts.append(text(x0 - 8, 256, 'Alba', 18, c['ink2']))
    return svg(w, h, 'Alba: a full-duplex voice model on a 4 GB laptop GPU',
               'Two voices on one timeline: the caller above the line and Alba below, overlapping where both talk.',
               ''.join(parts))


def answered_chart(c):
    load = lambda p: json.loads((RES / p).read_text(encoding='utf8'))
    rows = [('Cascade (first build)', load('tests/split/timing_scores.json')['all']['answered_share'], 'frozen'),
            ('Duplex, run 1', load('tests/duplex/timing_scores.json')['all']['answered_share'], 'frozen'),
            ('Duplex, run 2', load('calibration_alba-duplex2.json')['chosen']['answered_share'], 'validation')]
    people = load('tests/mock-human/timing_scores.json')['all']['answered_share']
    w, h, left, right, y0 = 900, 292, 220, 860, 104
    X = lambda v: left + v * (right - left)
    parts = [text(24, 34, 'The duplex model answers about 1 in 10 hand-offs; people answer 2 in 3', 18, c['ink'], 600),
             text(24, 58, "Turn hand-offs answered, scored from each system's sound on real two-person recordings", 13.5,
                  c['ink2'])]
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        parts.append(f'<line x1="{X(v):.1f}" y1="{y0 - 12}" x2="{X(v):.1f}" y2="{y0 + 3 * 44 - 8}" stroke="{c["grid"]}" '
                     f'stroke-width="1"/>')
        parts.append(text(X(v), y0 + 3 * 44 + 10, f'{v:.0%}', 12, c['ink2'], anchor='middle'))
    for k, (name, v, _) in enumerate(rows):
        y = y0 + k * 44
        fill = c['accent'] if k == len(rows) - 1 else c['grey']
        parts.append(text(left - 14, y + 16, name, 14, c['ink'], anchor='end'))
        parts.append(hbar(left, X(v), y, 22, fill))
        parts.append(text(X(v) + 8, y + 16, f'{v:.0%}', 14, c['ink'], 600))
    xp = X(people)
    parts.append(f'<line x1="{xp:.1f}" y1="{y0 - 18}" x2="{xp:.1f}" y2="{y0 + 3 * 44 - 8}" stroke="{c["ink2"]}" '
                 f'stroke-width="1.5"/>')
    parts.append(text(xp + 6, y0 - 22, f'People, same scorer: {people:.0%}', 13, c['ink2'], 600))
    parts.append(text(24, h - 14, 'Cascade and run 1: frozen test set, 15 real tracks (56 min). Run 2: 8 validation '
                      'conversations (28 min), listen setting 0. Data: results/.', 11.5, c['ink2']))
    return svg(w, h, 'Hand-offs answered by build',
               f'Cascade {rows[0][1]:.0%}, duplex run 1 {rows[1][1]:.0%}, duplex run 2 {rows[2][1]:.0%}; people '
               f'{people:.0%} on the same scorer.', ''.join(parts))


def speed_chart(c):
    load = lambda p: json.loads((RES / p).read_text(encoding='utf8'))
    a, f, k = load('fast_audio_check.json'), load('fast_frames_check.json'), load('fast_codec_check.json')
    before = a['library']['ms_per_step'] + f['model_ms'] + k['library']['ms_per_frame']
    after = a['fast-graph']['ms_per_step'] + f['fast-graph_ms'] + k['fast-graph']['ms_per_frame']
    w, h, left, right, y0, top = 900, 250, 220, 860, 104, 300.0
    X = lambda v: left + v / top * (right - left)
    parts = [text(24, 34, f'CUDA graphs cut the audio work per 80 ms step from {before:.0f} ms to {after:.0f} ms', 18,
                  c['ink'], 600),
             text(24, 58, 'Hearing, speech frame and sound, per step while talking, on the laptop GPU at 30 W', 13.5,
                  c['ink2'])]
    for v in (0, 100, 200, 300):
        parts.append(f'<line x1="{X(v):.1f}" y1="{y0 - 12}" x2="{X(v):.1f}" y2="{y0 + 2 * 44 - 8}" stroke="{c["grid"]}" '
                     f'stroke-width="1"/>')
        parts.append(text(X(v), y0 + 2 * 44 + 10, f'{v} ms', 12, c['ink2'], anchor='middle'))
    for i, (name, v, fill) in enumerate((('Library calls', before, c['grey']),
                                         ('Captured CUDA graphs', after, c['accent']))):
        y = y0 + i * 44
        parts.append(text(left - 14, y + 16, name, 14, c['ink'], anchor='end'))
        parts.append(hbar(left, X(v), y, 22, fill))
        parts.append(text(X(v) + 8, y + 16, f'{v:.0f} ms', 14, c['ink'], 600))
    xb = X(80)
    parts.append(f'<line x1="{xb:.1f}" y1="{y0 - 18}" x2="{xb:.1f}" y2="{y0 + 2 * 44 - 8}" stroke="{c["ink2"]}" '
                 f'stroke-width="1.5"/>')
    parts.append(text(xb + 6, y0 - 22, 'Real-time budget: 80 ms', 13, c['ink2'], 600))
    parts.append(text(24, h - 14, (f"Hearing {a['library']['ms_per_step']:.0f} to {a['fast-graph']['ms_per_step']:.1f} ms, "
                                   f"frames {f['model_ms']:.0f} to {f['fast-graph_ms']:.1f} ms, sound "
                                   f"{k['library']['ms_per_frame']:.0f} to {k['fast-graph']['ms_per_frame']:.1f} ms; "
                                   'each matches the library output. Data: results/fast_*_check.json.'), 11.5, c['ink2']))
    return svg(w, h, 'Audio work per step before and after CUDA graphs',
               f'{before:.1f} ms with library calls, {after:.1f} ms with captured CUDA graphs, against an 80 ms budget.',
               ''.join(parts))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for theme, c in THEMES.items():
        for name, make in (('banner', banner), ('answered', answered_chart), ('speed', speed_chart)):
            (OUT / f'{name}-{theme}.svg').write_text(make(c), encoding='utf8', newline='\n')
    print('wrote', sorted(p.name for p in OUT.glob('*.svg')))


if __name__ == '__main__':
    main()
