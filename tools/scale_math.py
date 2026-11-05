"""Back-of-envelope cost of training a Dessa-style speech-only model.

Rule of thumb: training compute ~ 6 x parameters x training positions.
Positions = hours of audio x 3600 x frames per second of the audio tokenizer.
Every input is printed next to its source or assumption.
  python -I scale_math.py
"""

FRAMES_PER_S = 12.5          # Moshi's Mimi codec (Defossez et al. 2024); Dessa's rate is unknown
H100_DENSE_BF16 = 989e12     # NVIDIA H100 SXM dense BF16 peak
GB200_RACK_DENSE = 180e15    # GB200 NVL72 dense BF16 (half the 360 PFLOPS sparse figure)
UTIL = 0.40                  # assumed utilisation for both
# RTX 3050 Laptop (GA107, 2048 CUDA cores, ~1.5 GHz at a 30-60 W limit):
# FP16 tensor throughput ~ 2048 cores x 2 x 1.5e9 x 4 (tensor speed-up) ~ 2.5e13 peak
RTX3050_FP16 = 2.5e13
LAPTOP_UTIL = 0.30
RENT_PER_H100_HOUR = (2.0, 3.0)   # typical 2026 on-demand range, USD (assumption)


def flops(params, hours, fps=FRAMES_PER_S):
    return 6 * params * hours * 3600 * fps


def fmt_time(seconds):
    for unit, s in (('years', 365 * 86400), ('days', 86400), ('hours', 3600), ('minutes', 60)):
        if seconds >= s:
            return f'{seconds / s:,.1f} {unit}'
    return f'{seconds:.0f} s'


CASES = [
    ('tiny proof of concept (dGSLM-like)', 100e6, 2_000),
    ('small model, open data scale', 300e6, 50_000),
    ('LFM3-Audio size', 2e9, 100_000),
    ('LFM3-Audio size, Dessa-scale data', 2e9, 1_000_000),
    ('Moshi size, Dessa-scale data', 7e9, 1_000_000),
    ('Moshi size, Moshi-scale data', 7e9, 7_000_000),
]

if __name__ == '__main__':
    print(f"{'case':38s} {'compute':>10s} {'this laptop':>16s} {'1x H100':>14s} {'GB200 rack':>12s} {'H100 rental':>16s}")
    for name, n, h in CASES:
        c = flops(n, h)
        laptop = c / (RTX3050_FP16 * LAPTOP_UTIL)
        h100 = c / (H100_DENSE_BF16 * UTIL)
        rack = c / (GB200_RACK_DENSE * UTIL)
        lo, hi = (h100 / 3600 * p for p in RENT_PER_H100_HOUR)
        print(f"{name:38s} {c:10.1e} {fmt_time(laptop):>16s} {fmt_time(h100):>14s} {fmt_time(rack):>12s} "
              f"{'$' + format(lo, ',.0f') + '-' + format(hi, ',.0f'):>16s}")
    # Inference: can a model run in real time on this laptop's 4 GB card?
    BW = 192e9   # RTX 3050 Laptop: 128-bit GDDR6 at 12 Gbps
    print('\nReal-time check on this laptop (each audio frame reads every weight once):')
    for name, n, bytes_per in (('LFM3-Audio-2B, 8-bit', 2.0e9, 1.0), ('Moshi 7B, 8-bit', 7.7e9, 1.0), ('Moshi 7B, 4-bit', 7.7e9, 0.56)):
        size = n * bytes_per
        fps = BW / size
        print(f"  {name:22s} weights {size / 1e9:4.1f} GB, fits in 4 GB: {'yes' if size < 3.3e9 else 'no '}, "
              f"bandwidth limit ~{fps:5.0f} steps/s vs {FRAMES_PER_S} needed")
