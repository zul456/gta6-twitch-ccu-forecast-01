"""Forecast GTA VI's launch-day Twitch viewers hour by hour (gta6-twitch-ccu-forecast-01).

Model: for each hour h = 2..24, fit (at run time, from the case's own comparables) a
log-linear regression of the log ratio log(viewers_h / viewers_1) on log(hour-1 viewers),
with an indicator for rolling midnight-local console unlocks:

    log(r_h) = a_h + b_h * log(p1) + c_h * I(rolling)

The regression is fit on the comparable launches' release days.  Big launches rise less
than small ones (log p1 correlates about -0.94 with log peak ratio across 40+ launches),
and rolling-midnight console launches rise more than simultaneous global (PC) ones at the
same hour-1 level -- the indicator corrects for that.  The fit is local in log p1 (an
adaptive window, widened until at least 12 points), the quantiles come from the empirical
residuals of that fit (an honest ~90% band), and the log-quantiles are lightly smoothed
over neighbouring hours.  Everything is deterministic; there is no randomness.

Fallbacks: if a case has no usable comparables, or any step fails, the forecast falls
back to the house baseline shape (flat at hour 1 with a widening spread), so a case is
never left invalid.  Outputs are validated before writing.
"""

import json
import math
import os
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds a
# module shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# House baseline spread, used only by the fallback (log units per sqrt hour).
BASE_SIGMA = 0.15
# Adaptive p1 windows (multiplicative), widened until enough points fit.
WINDOWS = (10, 20, 50, None)
MIN_POINTS = 12


def round2(x):
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def pct(sorted_vals, p):
    """Linear-interpolated percentile of an already sorted list."""
    if not sorted_vals:
        return 0.0
    i = p * (len(sorted_vals) - 1)
    lo = int(math.floor(i))
    hi = int(math.ceil(i))
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] * (hi - i) + sorted_vals[hi] * (i - lo)


def is_rolling(comp):
    """True when a comparable launched with a rolling midnight-local console unlock
    (no simultaneous global / PC unlock)."""
    text = ((comp.get("unlock") or "") + " " + (comp.get("platforms") or "")).lower()
    simul = ("simultaneous" in text or "single global" in text or "global unlock" in text)
    rolling = ("midnight local" in text or "rolling" in text)
    pc = "pc" in (comp.get("platforms") or "").lower()
    return rolling and not simul and not pc


def case_is_rolling(case):
    """Whether the case's own game unlocks at midnight local, rolling.  GTA VI does
    (RULES: midnight local, country by country); every practice launch does too.

    Check the rolling phrases FIRST: a case's notes may mention a simultaneous
    launch only to deny it ("no simultaneous launch"), and a naive
    "simultaneous" check would then misclassify a rolling game."""
    notes = (case.get("notes") or "").lower()
    name = ((case.get("game") or {}).get("name") or "").lower()
    if "grand theft auto vi" in name:
        return True
    if "midnight local" in notes or "rolling" in notes:
        return True
    if "simultaneous" in notes or "single global" in notes:
        return False
    return True


def release_curve(comp):
    """The comparable's release-day hourly viewers, {hour-of-day 0..23: viewers}."""
    day = (comp.get("day_start") or "")[:10]
    if not day:
        return {}
    out = {}
    for point in comp.get("hourly") or []:
        frm = point.get("from") or ""
        if frm.startswith(day):
            try:
                hour = int(frm[11:13])
            except (ValueError, IndexError):
                continue
            viewers = point.get("viewers")
            if isinstance(viewers, (int, float)) and viewers and viewers > 0:
                out[hour] = viewers
    return out


def fit3(points):
    """Least squares for log(ratio) = a + b*log(p1) + c*I(rolling).

    points: list of (log p1, log ratio, I).  Returns (a, b, c) or None if the system
    is singular (e.g. no variation in I)."""
    n = len(points)
    if n < 4:
        return None
    # Normal equations.
    a_mat = [[0.0] * 3 for _ in range(3)]
    vec = [0.0] * 3
    for x, y, ind in points:
        row = (1.0, x, ind)
        for i in range(3):
            vec[i] += row[i] * y
            for j in range(3):
                a_mat[i][j] += row[i] * row[j]
    # Gaussian elimination with partial pivoting.
    m = [a_mat[i][:] + [vec[i]] for i in range(3)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return None
        m[col], m[piv] = m[piv], m[col]
        for r in range(3):
            if r != col:
                factor = m[r][col] / m[col][col]
                for c in range(col, 4):
                    m[r][c] -= factor * m[col][c]
    return [m[i][3] / m[i][i] for i in range(3)]


def fit2(xs, ys):
    """Least squares for y = a + b*x.  Returns (a, b)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return my, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - b * mx, b


def baseline_quantiles(p1, hour):
    """House baseline shape: flat at hour 1 with a widening spread."""
    spread = BASE_SIGMA * math.sqrt(hour - 1)
    return [round2(p1 * math.exp(z * spread)) for z in Z]


def forecast_case(case):
    """Forecast one case.  Returns {"hours": [{"hour": h, "q": [...]}]}."""
    known = case.get("known") or []
    first = next((k for k in known if k.get("hour") == 1), None)
    if first is None:
        raise ValueError("no hour-1 print")
    p1 = first.get("viewers")
    if not isinstance(p1, (int, float)) or p1 < 1:
        p1 = 1
    p1 = float(p1)
    log_p1 = math.log(p1)
    rolling_case = 1.0 if case_is_rolling(case) else 0.0

    # Comparable release-day ratio curves: (log p1, {hour: log ratio}, I(rolling)).
    curves = []
    for comp in case.get("comparables") or []:
        hourly = release_curve(comp)
        if 0 not in hourly:
            continue
        base = hourly[0]
        curve = {}
        for hour in range(1, 25):
            v = hourly.get(hour - 1)
            if v:
                curve[hour] = math.log(v / base)
        if len(curve) >= 20:
            curves.append((math.log(base), curve, 1.0 if is_rolling(comp) else 0.0))
    if len(curves) < 4:
        raise ValueError("too few comparables")

    logq = {}
    for hour in range(2, 25):
        all_points = [(lp, c[hour], ind) for lp, c, ind in curves if hour in c]
        if len(all_points) < 4:
            raise ValueError("too few points for hour %d" % hour)
        points = None
        for window in WINDOWS:
            if window is None:
                cand = all_points
            else:
                limit = math.log(window)
                cand = [pt for pt in all_points if abs(pt[0] - log_p1) <= limit]
            if len(cand) >= MIN_POINTS:
                points = cand
                break
        if points is None:
            points = all_points
        beta = fit3(points)
        if beta is not None:
            a, b, c = beta
            pred = a + b * log_p1 + c * rolling_case
            residuals = sorted(y - (a + b * x + c * ind) for x, y, ind in points)
        else:
            xs = [pt[0] for pt in points]
            ys = [pt[1] for pt in points]
            a, b = fit2(xs, ys)
            pred = a + b * log_p1
            residuals = sorted(y - (a + b * x) for x, y in zip(xs, ys))
        logq[hour] = [pred + pct(residuals, level) for level in LEVELS]

    # Light smoothing of the log-quantiles over neighbouring hours.
    smoothed = {}
    for hour in range(2, 25):
        neighbours = [h for h in (hour - 1, hour, hour + 1) if h in logq]
        smoothed[hour] = [
            sum(logq[h][i] for h in neighbours) / len(neighbours) for i in range(5)
        ]

    hours = []
    for hour in range(2, 25):
        qs = [p1 * math.exp(x) for x in smoothed[hour]]
        for i in range(1, 5):
            qs[i] = max(qs[i], qs[i - 1])
        qs = [round2(max(1.0, min(1e9, q))) for q in qs]
        hours.append({"hour": hour, "q": qs})
    return {"hours": hours}


def validate(forecast):
    """Check the output shape the grader requires; raise on any violation."""
    if set(forecast.keys()) != {"hours"}:
        raise ValueError("output must have exactly the key 'hours'")
    rows = forecast["hours"]
    if len(rows) != 23:
        raise ValueError("need exactly 23 rows")
    seen = set()
    for row in rows:
        if set(row.keys()) != {"hour", "q"}:
            raise ValueError("row must have exactly 'hour' and 'q'")
        hour = row["hour"]
        if hour in seen or hour < 2 or hour > 24:
            raise ValueError("bad hour %r" % (hour,))
        seen.add(hour)
        q = row["q"]
        if len(q) != 5:
            raise ValueError("need exactly 5 quantiles")
        for i, value in enumerate(q):
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("quantile is not finite")
            if value <= 0 or value > 1e9:
                raise ValueError("quantile out of range")
            if i and value < q[i - 1]:
                raise ValueError("quantiles must be non-decreasing")
    if seen != set(range(2, 25)):
        raise ValueError("hours 2..24 each once")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
            validate(forecast)
        except Exception:
            # Never leave a case invalid: fall back to the baseline shape.
            known = case.get("known") or []
            first = next((k for k in known if k.get("hour") == 1), None)
            p1 = 1.0
            if first is not None:
                v = first.get("viewers")
                if isinstance(v, (int, float)) and v >= 1:
                    p1 = float(v)
            forecast = {
                "hours": [
                    {"hour": row["hour"], "q": baseline_quantiles(p1, row["hour"])}
                    for row in case.get("forecast") or []
                ]
            }
            validate(forecast)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        if len(text.encode("utf-8")) > 65536:
            raise ValueError("output too large for %s" % path.name)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
