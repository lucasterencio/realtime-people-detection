import time

class DwellTimeTracker:
    def __init__(self):
        self._entries = {}

    def update(self, names_in_frame):
        now = time.time()
        current = set(names_in_frame)

        for name in current:
            if name not in self._entries:
                self._entries[name] = now

        for name in list(self._entries.keys()):
            if name not in current:
                del self._entries[name]

    def get_dwell_time(self, name):
        if name in self._entries:
            return time.time() - self._entries[name]
        return 0.0

    def get_all_dwell_times(self):
        now = time.time()
        return {name: now - start for name, start in self._entries.items()}
