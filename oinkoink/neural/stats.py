import numpy as np


class ValueStats():
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
            self.total[k] += len(idx)
            self.correct[k] += len(np.equal(categories[idx], values[idx]).nonzero()[0])

    def categorise_predictions(self, preds):
        preds = preds * 3.0
        preds = np.floor(preds)
        preds = preds / 2.0
        return preds


class PriorStats():
    def __init__(self):
        self.n = 0
        self.total_loss = 0.0
        self.correct = 0

    @property
    def loss(self):
        return float(self.total_loss) / self.n

    @property
    def accuracy(self):
        return float(self.correct) / self.n

    def to_dict(self):
        dict_ = {'Average loss': self.loss,
                 'Accuracy': self.accuracy}
        return dict_

    def __repr__(self):
        x = "Average loss:  " + "{:.5f}".format(self.loss) + \
            "  Accuracy:  " + "{:.5f}".format(self.accuracy)

        return x

    def update(self, outputs, values, loss):
        self.n += len(values)
        self.total_loss += loss * len(values)

        output_best_move = np.argmax(outputs, axis=1)
        value_largest = np.amax(values, axis=1)
        # value_best_moves = [[x if i == l else 0 for x, i in enumerate(v)] for v, l in zip(values, value_largest)
        value_best_moves = []

        for largest, v in zip(value_largest, values):
            value_best_moves.append(np.where(v == largest)[0])

        for x, y in zip(output_best_move, value_best_moves):
            if x in y:
                self.correct += 1


class CombinedStats:
    def __init__(self):
        self.value_stats = ValueStats()
        self.prior_stats = PriorStats()

    @property
    def loss(self):
        return self.value_stats.loss + self.prior_stats.loss

    def update(self,
               value_outputs,
               values,
               value_loss,
               prior_outputs,
               priors,
               prior_loss):
        self.value_stats.update(value_outputs, values, value_loss)
        self.prior_stats.update(prior_outputs, priors, prior_loss)

    def to_dict(self):
        dict_ = {'prior '+k: v for k, v in self.prior_stats.to_dict().items()}
        dict_.update(self.value_stats.to_dict())
        return dict_

    def __repr__(self):
        return "{}\n{}".format(self.value_stats.__repr__(),
                               self.prior_stats.__repr__())

# a = np.array([[0,1], [1,1], [0,1], [0,1]])
# b = np.array([[1,1], [0,1], [0,1], [1,0]])
# p = PriorStats()
# p.update(a, b, 0.1)
# print(p)
