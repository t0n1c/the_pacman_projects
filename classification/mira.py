# mira.py
# -------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# Mira implementation
from .. import util

PRINT = True

class MiraClassifier:
    """
    Mira classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False
        self.C = 0.001
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."

        self.features = list(trainingData[0].keys()) # this could be useful for your code later...

        if (self.automaticTuning):
            Cgrid = [0.002, 0.004, 0.008]
        else:
            Cgrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

    def trainAndTune(self, data, labels, validation_data, validation_labels, Cgrid):
        """
        This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid,
        then store the weights that give the best accuracy on the val_data.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.
        """
        max_accuracy = float('-Inf')
        best_weights = None
        for c_bound in Cgrid:
            weights = self._train_classifier(data, labels, c_bound)
            accuracy = _get_accuracy(weights, validation_data, validation_labels)
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                best_weights = weights
        self.weights = best_weights


    def classify(self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses

    def _train_classifier(self, data, labels, c_bound):
        """Return a weight vector trained over the given data set and the upper bound
        (constant C in MIRA) of the step size (learning-rate like).

        Args:
            data: Training examples (features).
            labels: Training labels.
            c_bound: Constant C.

        """
        weights = self.weights.copy()
        for _ in range(self.max_iterations):
            for features,label in zip(data, labels):
                guessed_label = _predict_label(weights, features)
                if label != guessed_label:
                    f_prime = _scale_features(features, label, guessed_label, weights, c_bound)
                    weights[label] += f_prime
                    weights[guessed_label] -= f_prime
        return weights


def _get_accuracy(weights, val_data, val_labels):
    """Return the accuracy of a given vector over the validation set by computing the
    mean/percentage of hits (labels classified properly).

    Args:
        weights: A particular trained weight vector.
        val_data: Validation set (features).
        val_labels: Validation set (labels).
    """
    hits = [int(_predict_label(weights, f) == lab) for f,lab in zip(val_data,val_labels)]
    return sum(hits)/len(hits)


def _predict_label(weights, features):
    """Return the predicted label by computing the activation function for each weight and
    then taking the maximum.
    """
    return util.Counter([(lab, w*features) for lab,w in weights.items()]).argMax()


def _scale_features(features, label, predicted_label, weights, c_bound):
    """Return the feature vector step sized. It is intended to smooth the weight
    update in MIRA (minimum correcting update). Don't confuse with feature scaling.

    Args:
        features: Vector of features for a particular training example.
        label: Correct label corresponding to the features.
        predicted_label: Guessed label according to the current weights.
        weights: A particular trained weight vector.
        c_bound: constant C in MIRA.
    """
    wy = weights[label]
    wy_prime = weights[predicted_label]
    tau = _get_tau(features, wy, wy_prime, c_bound)
    return util.Counter([(pos, pix*tau) for pos,pix in features.items()])


def _get_tau(features, wy, wy_prime, upper_bound):
    """Return step size (tau) for the MIRA algorithm given an upper bound.

    Args:
        features: Vector of features for a particular training example.
        wy: Correct label corresponding to the features.
        wy_prime: Guessed label according to the current weights.
        upper_bound: constant C in MIRA.
    """
    tau = (wy_prime - wy)*features + 1.0
    tau = tau / (2*(features*features))
    return min(tau, upper_bound)





