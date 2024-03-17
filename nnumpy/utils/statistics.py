class RollingAverage:

    def __init__(self):
        self.total = 0
        self.count = 0

    def reset(self):
        self.total = 0
        self.count = 0

    def update(self, value):
        self.total += value
        self.count += 1

    @property
    def value(self):
        return self.total / self.count if self.count > 0 else 0
