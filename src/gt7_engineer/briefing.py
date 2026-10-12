"""The spoken lap update: built from race state, not from a model, so it is instant and exact.

Numbers are written as words because TTS engines read "1:32.4" and "0.3" unpredictably.
"""

from __future__ import annotations

from .state import NewLap, StateStore

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
         "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def say_int(n: int) -> str:
    """0..999 in words ('thirty-two'). Larger numbers fall back to digits."""
    if n < 0:
        return "minus " + say_int(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + (f"-{_ONES[n % 10]}" if n % 10 else "")
    if n < 1000:
        return _ONES[n // 100] + " hundred" + (f" {say_int(n % 100)}" if n % 100 else "")
    return str(n)


def say_num(x: float, decimals: int = 1) -> str:
    """1.7 -> 'one point seven'; 12.0 -> 'twelve'."""
    x = round(x, decimals)
    whole, frac = divmod(round(abs(x) * 10**decimals), 10**decimals)
    sign = "minus " if x < 0 else ""
    if frac == 0:
        return sign + say_int(int(whole))
    digits = str(int(frac)).zfill(decimals)
    return f"{sign}{say_int(int(whole))} point " + " ".join(_ONES[int(d)] for d in digits)


def say_lap_time(ms: int) -> str:
    """92400 -> 'one thirty-two point four'; 65050 -> 'one oh five point one'; 5200 -> 'five point two'."""
    tenths_total = (ms + 50) // 100
    minutes, rem = divmod(tenths_total, 600)
    seconds, tenth = divmod(rem, 10)
    if minutes == 0:
        return f"{say_int(seconds)} point {_ONES[tenth]}"
    sec = f"oh {_ONES[seconds]}" if seconds < 10 else say_int(seconds)
    return f"{_ONES[minutes] if minutes < 20 else str(minutes)} {sec} point {_ONES[tenth]}"


def say_delta(ms: int) -> str:
    """Gap to best lap: 'three tenths', 'one point two seconds', or '' when level."""
    tenths = (abs(ms) + 50) // 100
    if tenths == 0:
        return ""
    if tenths < 10:
        return f"{_ONES[tenths]} tenth" + ("" if tenths == 1 else "s")
    return f"{say_num(tenths / 10)} seconds"


def lap_update(store: StateStore, event: NewLap) -> str:
    """One short radio update for the lap that is just starting."""
    car = store.car
    rec = event.record
    parts: list[str] = []

    if car is not None:
        new_lap = car.lap
        parts.append(f"Lap {say_int(new_lap)}" + (f" of {say_int(car.total_laps)}" if car.total_laps else "") + ".")

    if rec.time_ms and rec.time_ms > 0:
        before = [l.time_ms for l in store.laps if l is not rec and l.time_ms > 0]
        best = min(before) if before else None
        text = say_lap_time(rec.time_ms)
        if best is None:
            parts.append(f"Last lap {text}.")
        elif rec.time_ms < best:
            d = say_delta(best - rec.time_ms)
            parts.append(f"New best, {text}" + (f", {d} up." if d else "."))
        else:
            d = say_delta(rec.time_ms - best)
            parts.append(f"Last lap {text}, " + (f"{d} off best." if d else "level with your best."))

    fpl = store.fuel_per_lap
    if car is not None and fpl:
        laps_of_fuel = car.fuel / fpl
        left = store.laps_remaining
        fuel = f"Fuel {say_num(laps_of_fuel)} laps" + (f", {say_int(left)} to go." if left is not None else ".")
        if left is not None:
            margin = laps_of_fuel - left
            if margin < 0:
                fuel += " Short, save fuel or box."
            elif margin < 1:
                fuel += " Tight."
        trend = store.burn_trend
        if trend and trend[1] in ("heavier", "lighter"):
            fuel += f" Burn {'rising' if trend[1] == 'heavier' else 'falling'}."
        parts.append(fuel)
    elif car is not None and car.fuel_capacity:
        parts.append(f"Fuel {say_int(round(car.fuel))} litres.")

    if store.position is not None and car is not None:
        text = f"Position {say_int(store.position)}"
        prev = store.position_at_lap_start.get(car.lap - 1)  # where you were when the last lap began
        if prev is not None and prev != store.position:
            text += f", {'up' if store.position < prev else 'down'} {say_int(abs(prev - store.position))}"
        parts.append(text + ".")
    else:  # the mode doesn't report position: use what the driver said, if recent
        said = store.latest_report("position")
        if said and car is not None and car.t - said.t < 4 * 90:
            parts.append(f"You said {said.value.split(' at lap')[0]}.")
    return " ".join(parts)
