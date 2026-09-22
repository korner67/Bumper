import argparse
import asyncio
import json
import math
import os
import random
import string
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import discord

BASE_DIR = Path(__file__).resolve().parent

CONFIG = json.loads((BASE_DIR / "config.json").read_text(encoding="utf-8"))
TOKEN = os.environ.get("ELEMENTAL_TOKEN", "")
STATE_FILE = str(BASE_DIR / "suite_state.json")
PROFILE_FILE = str(BASE_DIR / "suite_profile.json")


def _floored_lognormal(median, sigma, floor, rng=None, max_tries=20):
    r = rng or random
    m = math.log(median)
    for _ in range(max_tries):
        v = math.exp(r.gauss(m, sigma))
        if v >= floor:
            return v
    return floor


def reaction_time(rng=None):
    return _floored_lognormal(0.45, 0.35, 0.32, rng)


def lognormal(median, sigma=0.6, cap=8.0, max_tries=20):
    m = math.log(median)
    for _ in range(max_tries):
        v = math.exp(random.gauss(m, sigma))
        if v <= cap:
            return v
    return cap


class AccountProfile:
    def __init__(self, seed: str):
        rng = random.Random(seed)
        self.gap_mu = rng.uniform(1.5, 4.0)
        self.gap_sigma = rng.uniform(0.5, 1.0)
        self.no_typing_rate = rng.uniform(0.10, 0.35)
        self.cancel_typing_rate = rng.uniform(0.03, 0.15)
        self.resume_drift_min = rng.uniform(3, 25)
        self.sleep_start = rng.randint(0, 2)
        self.sleep_end = rng.randint(6, 9)
        self.rt_mu = rng.uniform(0.35, 0.65)
        self.rt_sigma = rng.uniform(0.22, 0.50)
        self.rt_floor = rng.uniform(0.25, 0.42)

    def next_gap_minutes(self):
        return math.exp(random.gauss(math.log(self.gap_mu), self.gap_sigma))

    def is_sleeping(self, now):
        h = now.hour
        if self.sleep_start < self.sleep_end:
            return self.sleep_start <= h < self.sleep_end
        return h >= self.sleep_start or h < self.sleep_end

    def rt(self):
        return _floored_lognormal(self.rt_mu, self.rt_sigma, self.rt_floor)


def load_profile():
    p = Path(PROFILE_FILE)
    if p.exists():
        return AccountProfile(json.loads(p.read_text(encoding="utf-8"))["seed"])
    seed = "".join(random.choices(string.ascii_letters + string.digits, k=16))
    p.write_text(json.dumps({"seed": seed}), encoding="utf-8")
    return AccountProfile(seed)


def _read_last_bump():
    p = Path(STATE_FILE)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        ts = data.get("last_bump")
        if ts is None:
            return None
        if isinstance(ts, (int, float)):
            return float(ts)
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def _write_last_bump(ts):
    p = Path(STATE_FILE)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps({"last_bump": ts}),
                   encoding="utf-8")
    os.replace(tmp, p)


class Suite(discord.Client):

    MAX_FAILURES = 5
    FAILURE_RETRY_DELAY = 30

    def __init__(self, level: int, profile: AccountProfile, max_sends: int):
        super().__init__()
        self.level = level
        self.profile = profile
        self.max_sends = max_sends
        self.sends = 0
        self.last_bump = None
        self._cycle_task = None
        self._finished = False
        self._load_state()

    def _load_state(self):
        self.last_bump = _read_last_bump()

    def _save_state(self):
        _write_last_bump(self.last_bump)

    async def _rt(self):
        if self.level < 2:
            return
        if self.level < 4:
            await asyncio.sleep(reaction_time())
        else:
            await asyncio.sleep(self.profile.rt())

    async def on_ready(self):
        user = self.user
        print(f"Connected as {user} (ID: {user.id}) â€” level {self.level}")

        if self._finished:
            return

        await self._rt()

        if self.level == 1:
            await self.change_presence(
                activity=discord.Activity(
                    type=discord.ActivityType.watching,
                    name=CONFIG.get("activity_name", "")),
                status=discord.Status.online)

        if self._cycle_task is None or self._cycle_task.done():
            self._cycle_task = asyncio.create_task(self.cycle())
            self._cycle_task.add_done_callback(self._on_cycle_done)

    def _on_cycle_done(self, task):
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            print(f"cycle terminated with error: {exc!r}")

    async def cycle(self):
        try:
            await self.wait_until_ready()

            channel = self.get_channel(CONFIG["channel_id"])
            if not channel:
                print("Channel not found")
                return

            failures = 0
            while not self.is_closed() and self.sends < self.max_sends:
                now_local = datetime.now(timezone.utc).astimezone()
                if self.level >= 4 and self.profile.is_sleeping(now_local):
                    wake = now_local.replace(hour=self.profile.sleep_end,
                                             minute=0, second=0, microsecond=0)
                    wake += timedelta(minutes=random.uniform(0, 90))
                    if wake <= now_local:
                        wake += timedelta(days=1)
                    print(f"sleeping until {wake:%H:%M}")
                    await asyncio.sleep((wake - now_local).total_seconds())
                    continue

                await self._resume_wait()

                now_local = datetime.now(timezone.utc).astimezone()
                if self.level >= 4 and self.profile.is_sleeping(now_local):
                    continue

                await self._do_typing(channel)

                if await self._do_send(channel):
                    failures = 0
                    if self.sends >= self.max_sends:
                        break
                    await self._inter_gap()
                else:
                    failures += 1
                    if failures >= self.MAX_FAILURES:
                        print("too many consecutive failures, stopping")
                        break
                    await asyncio.sleep(self.FAILURE_RETRY_DELAY)

            print(f"Done. {self.sends} sends at level {self.level}.")
        finally:
            try:
                await self._finish()
            except Exception as e:
                print(f"_finish failed: {e!r}")

    async def _finish(self):
        if self._finished:
            return
        self._finished = True
        await self.close()

    async def _do_typing(self, channel):
        if self.level == 1:
            async with channel.typing():
                await asyncio.sleep(random.uniform(2, 7))
            await asyncio.sleep(random.uniform(0.5, 2.5))
        elif self.level >= 2:
            r = random.random()
            if r < self.profile.cancel_typing_rate:
                async with channel.typing():
                    await asyncio.sleep(random.uniform(1, 3))
                await asyncio.sleep(random.uniform(3, 20))
            elif r < self.profile.cancel_typing_rate + self.profile.no_typing_rate:
                pass
            else:
                async with channel.typing():
                    await asyncio.sleep(random.uniform(0.8, 2.5))
                await asyncio.sleep(lognormal(1.2))

    async def _do_send(self, channel):
        try:
            await self._rt()

            if self.level <= 2:
                await channel.send("/bump")
            else:
                await self._send_real_interaction(channel)
        except Exception as e:
            print(f"send rejected: {e!r}")
            return False

        self.sends += 1
        self.last_bump = time.time()
        print(f"[{datetime.now(timezone.utc).astimezone():%H:%M:%S}] send #{self.sends}")
        try:
            self._save_state()
        except OSError as e:
            print(f"state not saved: {e!r}")
            return False
        return True

    @staticmethod
    def _check_response(resp, what):
        status = getattr(resp, "status_code", None)
        if status is not None and status >= 400:
            raise RuntimeError(f"{what}: HTTP {status}")
        return resp

    async def _send_real_interaction(self, channel):
        if hasattr(channel, "application_commands"):
            cmds = await channel.application_commands()
            app_id = str(CONFIG["target_application_id"])
            bump = next((c for c in cmds
                         if c.name == "bump" and str(c.application_id) == app_id),
                        None)
            if bump is None:
                raise LookupError("comando 'bump' non trovato")
            await self._rt()
            await bump(channel=channel)
            return

        r = self._check_response(
            await self.http.get(
                f"/guilds/{CONFIG['guild_id']}/applications/"
                f"{CONFIG['target_application_id']}/commands"),
            "get commands")
        data = r.json()
        if asyncio.iscoroutine(data):
            data = await data
        bump = next((c for c in data if c["name"] == "bump"), None)
        if bump is None:
            raise LookupError("comando 'bump' non trovato")
        await self._rt()
        self._check_response(
            await self.http.post("/interactions", json={
                "type": 2,
                "application_id": CONFIG["target_application_id"],
                "channel_id": CONFIG["channel_id"],
                "session_id": getattr(self, "session_id", None),
                "nonce": str(random.getrandbits(64)),
                "data": {"id": bump["id"], "name": "bump", "options": []},
            }),
            "post interaction")

    async def _resume_wait(self):
        if not self.last_bump:
            return
        elapsed = time.time() - self.last_bump
        remaining = CONFIG["min_interval"] - elapsed
        if remaining > 0:
            drift = random.uniform(0, self.profile.resume_drift_min * 60) \
                    if self.level >= 4 else 0
            print(f"resuming in {remaining + drift:.0f}s")
            await asyncio.sleep(remaining + drift)

    async def _inter_gap(self):
        if self.level == 1:
            base = random.randint(CONFIG["min_interval"], CONFIG["max_interval"])
            jitter = random.randint(-CONFIG["jitter_range"], CONFIG["jitter_range"])
            await asyncio.sleep(max(base + jitter, CONFIG["min_interval"]))
        else:
            await asyncio.sleep(self.profile.next_gap_minutes() * 60)


LEVELS = {
    1: "baseline",
    2: "floor",
    3: "no-presence",
    4: "per-account",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, choices=sorted(LEVELS), default=1)
    ap.add_argument("--max-sends", type=int, default=20)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for lv, desc in LEVELS.items():
            print(f"  level {lv}: {desc}")
        return

    if args.max_sends < 1:
        ap.error("--max-sends deve essere >= 1")

    if not TOKEN:
        raise SystemExit("Set ELEMENTAL_TOKEN")

    profile = load_profile()

    bot = Suite(args.level, profile, args.max_sends)
    try:
        bot.run(TOKEN)
    except discord.LoginFailure:
        print("Invalid token")
    except KeyboardInterrupt:
        print("Interrotto")


if __name__ == "__main__":
    main()
