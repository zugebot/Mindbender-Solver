# Jerrin Shirks

# native imports
import json
import time
from pathlib import Path


class MindBenderStats:
    def __init__(self, stats_file):
        self.stats_file = Path(stats_file)
        self.data = self._load_stats()

    def _now(self):
        return time.time()

    def _default_stats(self):
        return {
            "puzzles": {},
            "worlds_any3": {},
            "worlds_all5": {},
        }

    def _normalize_bucket(self, bucket):
        if not isinstance(bucket, dict):
            return {}

        normalized = {}

        for key, entry in bucket.items():
            if not isinstance(entry, dict):
                continue

            count = entry.get("count", 0)
            latest_seconds = entry.get("latest_seconds")
            best_seconds = entry.get("best_seconds")
            average_seconds = entry.get("average_seconds")
            best_run = entry.get("best_run")

            runs = entry.get("runs", [])
            total_seconds = entry.get("total_seconds")

            # Migrate old format that stored every run forever.
            if isinstance(runs, list) and runs:
                valid_run_times = []
                best_run_from_runs = None

                for run in runs:
                    if not isinstance(run, dict):
                        continue

                    elapsed = run.get("elapsed_seconds")
                    if isinstance(elapsed, (int, float)):
                        elapsed = float(elapsed)
                        valid_run_times.append(elapsed)

                        if best_run_from_runs is None or elapsed < best_run_from_runs["elapsed_seconds"]:
                            best_run_from_runs = dict(run)

                if valid_run_times:
                    count = len(valid_run_times)
                    latest_seconds = float(valid_run_times[-1])
                    best_seconds = float(min(valid_run_times))
                    total_seconds = float(sum(valid_run_times))
                    average_seconds = total_seconds / count

                    if best_run_from_runs is not None:
                        best_run = best_run_from_runs

            if not isinstance(count, int) or count < 0:
                count = 0

            if isinstance(total_seconds, (int, float)):
                total_seconds = float(total_seconds)
            else:
                if count > 0 and isinstance(average_seconds, (int, float)):
                    total_seconds = float(average_seconds) * count
                else:
                    total_seconds = 0.0

            if isinstance(latest_seconds, (int, float)):
                latest_seconds = float(latest_seconds)
            else:
                latest_seconds = None

            if isinstance(best_seconds, (int, float)):
                best_seconds = float(best_seconds)
            else:
                best_seconds = None

            if isinstance(average_seconds, (int, float)):
                average_seconds = float(average_seconds)
            else:
                average_seconds = (total_seconds / count) if count > 0 else None

            if count == 0:
                latest_seconds = None
                best_seconds = None
                average_seconds = None
                total_seconds = 0.0
                best_run = None

            normalized[key] = {
                "count": count,
                "latest_seconds": latest_seconds,
                "best_seconds": best_seconds,
                "average_seconds": average_seconds,
                "total_seconds": total_seconds,
                "best_run": best_run if isinstance(best_run, dict) else None,
            }

        return normalized

    def _load_stats(self):
        default_stats = self._default_stats()

        try:
            if self.stats_file.exists():
                with self.stats_file.open("r", encoding="utf-8") as infile:
                    loaded = json.load(infile)

                if not isinstance(loaded, dict):
                    return default_stats

                normalized = {}
                for key in default_stats:
                    normalized[key] = self._normalize_bucket(loaded.get(key, {}))

                return normalized
        except Exception:
            pass

        return default_stats

    def save(self):
        try:
            self.stats_file.parent.mkdir(parents=True, exist_ok=True)
            with self.stats_file.open("w", encoding="utf-8") as outfile:
                json.dump(self.data, outfile, indent=2, sort_keys=True)
        except Exception:
            pass

    def _append_stat(self, category, key, elapsed_seconds, extra=None):
        if extra is None:
            extra = {}

        elapsed_seconds = float(elapsed_seconds)

        if key not in self.data[category]:
            self.data[category][key] = {
                "count": 0,
                "best_seconds": None,
                "latest_seconds": None,
                "average_seconds": None,
                "total_seconds": 0.0,
                "best_run": None,
            }

        entry = self.data[category][key]

        old_count = entry.get("count", 0)
        old_total = entry.get("total_seconds", 0.0)

        if not isinstance(old_count, int) or old_count < 0:
            old_count = 0
        if not isinstance(old_total, (int, float)):
            old_total = 0.0

        new_count = old_count + 1
        new_total = float(old_total) + elapsed_seconds
        new_average = new_total / new_count

        entry["count"] = new_count
        entry["latest_seconds"] = elapsed_seconds
        entry["total_seconds"] = new_total
        entry["average_seconds"] = new_average

        current_best = entry.get("best_seconds")
        if current_best is None or elapsed_seconds < current_best:
            entry["best_seconds"] = elapsed_seconds
            best_run = {
                "elapsed_seconds": elapsed_seconds,
                "timestamp": int(self._now()),
            }
            best_run.update(extra)
            entry["best_run"] = best_run

        self.save()

    def record_puzzle(self, world_number, level_number, puzzle_index, elapsed_seconds):
        key = f"world_{world_number}_puzzle_{level_number}"
        self._append_stat(
            "puzzles",
            key,
            elapsed_seconds,
            extra={
                "world_number": world_number,
                "level_number": level_number,
                "puzzle_index": puzzle_index,
            }
        )

    def record_world_any3(self, world_number, solved_count_when_leaving, reason, elapsed_seconds):
        key = f"world_{world_number}"
        self._append_stat(
            "worlds_any3",
            key,
            elapsed_seconds,
            extra={
                "world_number": world_number,
                "solved_count_when_leaving": solved_count_when_leaving,
                "reason": reason,
            }
        )

    def record_world_all5(self, world_number, solved_count_when_leaving, reason, elapsed_seconds):
        key = f"world_{world_number}"
        self._append_stat(
            "worlds_all5",
            key,
            elapsed_seconds,
            extra={
                "world_number": world_number,
                "solved_count_when_leaving": solved_count_when_leaving,
                "reason": reason,
            }
        )

    def get_best_seconds(self, category, key):
        entry = self.data.get(category, {}).get(key)
        if not entry:
            return None
        return entry.get("best_seconds")

    def get_average_seconds(self, category, key):
        entry = self.data.get(category, {}).get(key)
        if not entry:
            return None
        return entry.get("average_seconds")

    def get_count(self, category, key):
        entry = self.data.get(category, {}).get(key)
        if not entry:
            return 0
        return entry.get("count", 0)

    @staticmethod
    def format_elapsed(seconds):
        if seconds is None:
            return "--:--.---"

        total_seconds = float(seconds)
        minutes = int(total_seconds // 60)
        secs = total_seconds - (minutes * 60)
        return f"{minutes:02d}:{secs:05.3f}"