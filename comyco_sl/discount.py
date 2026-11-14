import numpy as np

START = 0.9
STEP = 0.05
TOL = 0.01  # relative change of the window mean that counts as better or worse
FLOOR = 0.4
WINDOW = 5  # validations per decision


class Discount:
    # expert bandwidth scale, hill climbing on the validation score: keep going when the window mean is better by
    # TOL, turn around when it is worse by TOL, stay put otherwise
    def __init__(self):
        self.value, self.direction, self.prev, self.scores = START, -1, None, []

    def update(self, score):
        self.scores.append(score)
        if len(self.scores) < WINDOW:
            return False
        mean = float(np.mean(self.scores))
        first = self.prev is None
        change = 0.0 if first else mean / self.prev - 1.0
        self.prev, self.scores = mean, []
        if change < -TOL:
            self.direction = -self.direction
        elif not first and change <= TOL:
            return False
        old = self.value
        self.value = round(float(np.clip(self.value + self.direction * STEP, FLOOR, 1.0)), 3)
        return self.value != old
