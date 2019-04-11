import numpy as np


class Stats():
    def __init__(self):
        self.n = 0
        self.average_value = 0.0
        self.total_loss = 0.0
        self.smallest = 1.0
        self.largest = 0.0
        self.correct = {i: 0 for i in [0.0, 0.5, 1.0]}
        self.total = {i: 0 for i in [0.0, 0.5, 1.0]}

    @property
    def loss(self):
        return float(self.total_loss) / self.n

    @property
    def accuracy(self):
        return float(sum(self.correct.values())) / self.n

    @property
    def average(self):
        return self.average_value / self.n

    def to_dict(self):
        dict_ = {'Average loss': self.loss,
                 'Accuracy': self.accuracy,
                 'Smallest': self.smallest,
                 'Largest': self.largest,
                 'Average': self.average}
        dict_['correct'] = {}
        for k in self.correct:
            dict_['correct'][k] = (self.total[k], self.correct[k])

        return dict_

    def __repr__(self):
        x = "Average loss:  " + "{:.5f}".format(self.loss) + \
            "  Accuracy:  " + "{:.5f}".format(self.accuracy) + \
            "  Smallest:  " + "{:.5f}".format(self.smallest) + \
            "  Largest:  " + "{:.5f}".format(self.largest) + \
            "  Average:  " + "{:.5f}".format(self.average) + \
            "\nCategory, # Members, # Correct Predictions:"

        for k in self.correct:
            x += "  ({}, {}, {})".format(
                k,
                self.total[k],
                self.correct[k])
        return x

    def update(self, outputs, values, loss):
        self.n += len(values)
        self.average_value += np.sum(outputs)
        self.total_loss += loss * len(values)
        self.smallest = min(self.smallest, np.min(outputs).item())
        self.largest = max(self.largest, np.max(outputs).item())

        categories = self.categorise_predictions(outputs)

        for k in self.correct:
            idx = np.where(values == k)[0]
            # print(np.equal(categories[idx], values[idx]).nonzero()[0])
            self.total[k] += len(idx)
            self.correct[k] += len(np.equal(categories[idx], values[idx]).nonzero()[0])

    def categorise_predictions(self, preds):
        preds = preds * 3.0
        preds = np.floor(preds)
        preds = preds / 2.0
        return preds
