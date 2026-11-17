"""gta6-twitch-ccu-forecast-01 entry: an empirical comparable-launch forecaster.

The idea: a launch day's Twitch curve, measured as the log ratio to its own
first hour, is a fairly stable animal. Launches that were already live around
the world by 00:00 UTC (a rolling midnight unlock shows up as a large last
pre-release hour) and that opened at a similar scale trace similar days: a
high first hour, a late-morning dip, a long evening recovery. So this program
reads the case it is given, builds the release-day curve of every comparable
launch in it, weights those launches by how closely their opening scale and
their pre-release run-in match this case's, and takes weighted quantiles of
their log ratios, hour by hour. The empirical quantiles are blended with a
weighted normal fit at the tails (a few dozen launches make raw 5% and 95%
quantiles jumpy), shrunk slightly toward the median, and nudged by a small
constant; the blend weights and kernel widths were set by leave-one-out
backtests over the comparable pool (each launch forecast from the others and
scored with the contest's own pinball loss), which put the model well under
the persistence baseline on the launches that resemble this one.

Everything is computed from the case file at run time: no network, no
randomness, no clock, so two runs write byte-identical forecasts. If
anything about a case is unusable, the program falls back to a
persistence-shaped forecast rather than risk an invalid file.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

STD_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
NORM = NormalDist()

# Kernel widths and blend constants, set by leave-one-out backtests.
SIG_SCALE = 1.2    # kernel width on log opening scale
SIG_LAST = 0.7     # kernel width on log last-pre-hour / first-hour ratio
POOL = 0.2         # weight of the neighbouring hours in each hour's sample
BETA_TAIL = 0.35   # normal-fit blend at the outermost levels
BETA_MID = 0.1     # normal-fit blend at the inner levels
GAMMA = 0.95       # spread shrink toward the median
DELTA = -0.02      # constant log shift of every quantile


def parse_ts(text):
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def round2(x):
    """Round half up to 2 decimals; whole numbers come back as int."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if float(value).is_integer() else value


def hourly_series(points, start, key):
    """Map hour-offset-from-start -> viewers for a list of hourly points."""
    out = {}
    for p in points or []:
        stamp = p.get("from") or p.get("at")
        if stamp is None:
            continue
        idx = int((parse_ts(stamp) - start).total_seconds() // 3600)
        out[idx] = max(p.get(key, 0) or 0, 0)
    return out


def build_curve(points, start):
    """A comparable's release day: (viewers[24], last pre-day hour viewers).

    Missing day hours are filled by log interpolation; returns None when the
    launch has no usable first hour or too sparse a day.
    """
    series = hourly_series(points, start, "viewers")
    day = [series.get(i) for i in range(24)]
    if day[0] is None or day[0] < 50:
        return None
    present = [i for i in range(24) if day[i] is not None]
    if len(present) < 18:
        return None
    for i in range(24):
        if day[i] is None:
            lo = max([j for j in present if j < i], default=None)
            hi = min([j for j in present if j > i], default=None)
            if lo is None:
                day[i] = day[hi]
            elif hi is None:
                day[i] = day[lo]
            else:
                f = (i - lo) / (hi - lo)
                day[i] = math.exp(math.log(max(day[lo], 1)) * (1 - f)
                                  + math.log(max(day[hi], 1)) * f)
    pre_last = 0
    for i in range(-6, 0):
        if i in series:
            pre_last = series[i]
    return day, pre_last


def wquantile(pairs, tau):
    """Weighted quantile with linear interpolation on the cumulative weight."""
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    if total <= 0:
        return pairs[len(pairs) // 2][0]
    target = tau * total
    acc = 0.0
    for i, (x, w) in enumerate(pairs):
        acc += w
        if acc >= target:
            if i == 0:
                return x
            prev_x = pairs[i - 1][0]
            prev_acc = acc - w
            if acc == prev_acc:
                return x
            frac = (target - prev_acc) / (acc - prev_acc)
            return prev_x + frac * (x - prev_x)
    return pairs[-1][0]


def model_quantiles(case, levels):
    """The comparable-weighted log-ratio quantiles for one case.

    Returns {hour: [log quantiles]} or None when the case carries no usable
    comparable pool.
    """
    start = parse_ts(case["day_start"])
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        return None
    p1 = max(1, first["viewers"])

    target_hist = case.get("history_hourly") or case.get("history") or []
    t_series = hourly_series(target_hist, start, "viewers")
    t_pre_last = 0
    for i in range(-6, 0):
        if i in t_series:
            t_pre_last = t_series[i]
    ln_p1 = math.log(p1)
    ln_last = math.log(max(t_pre_last, 1) / p1)

    pool = []
    for comp in case.get("comparables") or []:
        try:
            built = build_curve(comp.get("hourly"), parse_ts(comp["day_start"]))
        except Exception:
            built = None
        if built is None:
            continue
        day, pre_last = built
        cp1 = max(day[0], 1)
        d2 = ((ln_p1 - math.log(cp1)) / SIG_SCALE) ** 2
        d2 += ((ln_last - math.log(max(pre_last, 1) / cp1)) / SIG_LAST) ** 2
        weight = math.exp(-0.5 * d2) + 1e-4
        ratios = {h: math.log(max(day[h - 1], 1) / cp1) for h in range(2, 25)}
        pool.append((ratios, weight))
    if len(pool) < 5:
        return None

    med_pos = min(range(len(levels)), key=lambda i: abs(levels[i] - 0.5))
    out = {}
    for row in case["forecast"]:
        h = row["hour"]
        samples = []
        for ratios, w in pool:
            for dh, pw in ((-1, POOL), (0, 1.0), (1, POOL)):
                hh = h + dh
                if 2 <= hh <= 24:
                    samples.append((ratios[hh], w * pw))
        total_w = sum(w for _, w in samples)
        mu = sum(x * w for x, w in samples) / total_w
        var = sum(w * (x - mu) ** 2 for x, w in samples) / total_w
        sd = math.sqrt(max(var, 1e-6))
        qs = []
        for pos, tau in enumerate(levels):
            emp = wquantile(samples, tau)
            beta = BETA_TAIL if pos in (0, len(levels) - 1) else BETA_MID
            qs.append((1 - beta) * emp + beta * (mu + NORM.inv_cdf(tau) * sd))
        med = qs[med_pos]
        out[h] = [med + GAMMA * (q - med) + DELTA for q in qs]
    return out


def persistence_quantiles(case, levels):
    """Fallback: hour 1's print for every hour, random-walk log spread."""
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    p1 = max(1, first["viewers"]) if first else 1
    return {row["hour"]: [NORM.inv_cdf(tau) * 0.15 * math.sqrt(row["hour"] - 1)
                           for tau in levels]
            for row in case["forecast"]}, p1


def forecast_case(case):
    levels = case.get("levels") or STD_LEVELS
    if case.get("forecast"):
        rows = case["forecast"]
    else:
        rows = [{"hour": h} for h in range(2, 25)]
    case = dict(case, forecast=rows)
    first = next((k for k in case.get("known") or [] if k["hour"] == 1), None)
    p1 = max(1, first["viewers"]) if first else 1
    log_qs = None
    try:
        log_qs = model_quantiles(case, levels)
    except Exception:
        log_qs = None
    if log_qs is None:
        log_qs, _ = persistence_quantiles(case, levels)
    hours = []
    for row in rows:
        h = row["hour"]
        qs = []
        prev = 0.0
        for lr in log_qs[h]:
            q = p1 * math.exp(lr)
            q = min(max(q, 0.01), 1e9)
            q = max(q, prev)
            qs.append(round2(q))
            prev = qs[-1]
        hours.append({"hour": h, "q": qs})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
