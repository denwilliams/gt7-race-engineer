from __future__ import annotations

from ..bus import IGNORE, MENTION, URGENT
from ..state import NewLap
from .base import Analyst


def fuel_rule(ctx: dict) -> str:
    margin = ctx.get("margin_laps")
    if ctx.get("burn_trend") == "heavier" and margin is not None and 0 <= margin < 2:
        return MENTION
    if margin is not None:
        if margin < 0:
            return URGENT if ctx["laps_of_fuel"] < 2 else MENTION
        return IGNORE if margin > 1 else MENTION
    return URGENT if ctx["laps_of_fuel"] < 1.5 else IGNORE


class FuelAgent(Analyst):
    name = "fuel"
    topic = "fuel"
    interval = 5.0
    instructions = (
        "You assess fuel for a race car. margin_laps is laps of fuel left minus laps still to run "
        "(negative means the car cannot finish without pitting). laps_of_fuel under 2 is urgent. "
        "burn_trend says whether fuel use per lap is getting 'heavier' or 'lighter' (or 'steady'); a heavier trend "
        "with a thin margin is worth mentioning. "
        "Say 'ignore' when comfortable, 'mention' when worth a heads-up, 'urgent' when the driver must act."
    )
    _last_level: str | None = None

    def _calc(self) -> tuple[dict, str] | None:
        fpl = self.store.fuel_per_lap
        c = self.store.car
        if not fpl or not c:
            return None
        laps_of_fuel = c.fuel / fpl
        ctx = {"fuel_l": round(c.fuel, 1), "fuel_per_lap_l": round(fpl, 2), "laps_of_fuel": round(laps_of_fuel, 1)}
        last = self.store.last_lap_burn
        if last:
            ctx["last_lap_burn_l"] = round(last, 2)
            ctx["laps_of_fuel_at_last_burn"] = round(c.fuel / last, 1)
        if (trend := self.store.burn_trend) is not None:
            ctx["burn_change_per_lap_l"] = round(trend[0], 2)
            ctx["burn_trend"] = trend[1]
        left = self.store.laps_remaining
        if left is not None:
            ctx["laps_remaining"] = left
            ctx["margin_laps"] = round(laps_of_fuel - left, 1)
            if ctx["margin_laps"] < 0:
                msg = f"Fuel: {ctx['laps_of_fuel']} laps left but {left} to run. Short by {-ctx['margin_laps']} laps, need to save or pit."
            else:
                msg = f"Fuel: {ctx['laps_of_fuel']} laps left, {left} to run, margin {ctx['margin_laps']} laps."
        else:
            msg = f"Fuel: {ctx['laps_of_fuel']} laps remaining at {ctx['fuel_per_lap_l']} L per lap."
        if ctx.get("burn_trend") in ("heavier", "lighter"):
            msg += (
                f" Burn is getting {ctx['burn_trend']}: {ctx['last_lap_burn_l']} L last lap vs "
                f"{ctx['fuel_per_lap_l']} L average, {ctx['laps_of_fuel_at_last_burn']} laps of fuel at the last lap's rate."
            )
        return ctx, msg

    def measure(self):
        # Between laps nothing new is learned; per-lap handler does the work.
        # Only watch for a sudden drop into the urgent zone.
        r = self._calc()
        if r and r[0]["laps_of_fuel"] < 1.5:
            return r
        return None

    def measure_lap(self, lap: NewLap):
        return self._calc()
