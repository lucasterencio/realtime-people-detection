import time
import config


class DwellTimeTracker:
    def __init__(self):
        self._entries = {}
        self._pending = {}

    def update(self, names_in_frame):
        now = time.time()
        current = set(names_in_frame)
        departed = []

        for name in current:
            if name in self._pending:
                start, _ = self._pending[name]
                self._entries[name] = start
                del self._pending[name]
            elif name not in self._entries:
                self._entries[name] = now

        for name in list(self._entries.keys()):
            if name not in current:
                self._pending[name] = (self._entries[name], now)
                del self._entries[name]

        for name in list(self._pending.keys()):
            start, left_at = self._pending[name]
            if now - left_at > config.DWELL_TOLERANCE_SECONDS:
                duration = left_at - start
                departed.append((name, duration, start, left_at))
                del self._pending[name]

        return departed

    def get_dwell_time(self, name):
        if name in self._entries:
            return time.time() - self._entries[name]
        if name in self._pending:
            start, left_at = self._pending[name]
            return left_at - start
        return 0.0

    def get_all_dwell_times(self):
        now = time.time()
        result = {}
        for name, start in self._entries.items():
            result[name] = now - start
        for name, (start, left_at) in self._pending.items():
            result[name] = left_at - start
        return result

    def flush(self):
        now = time.time()
        remaining = []
        for name in list(self._entries.keys()):
            duration = now - self._entries[name]
            remaining.append((name, duration, self._entries[name], now))
            del self._entries[name]
        for name in list(self._pending.keys()):
            start, left_at = self._pending[name]
            duration = left_at - start
            remaining.append((name, duration, start, left_at))
            del self._pending[name]
        return remaining
