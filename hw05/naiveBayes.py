import util
import classificationMethod
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.

  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "naivebayes"
    self.bestk = 1 # this is the smoothing parameter, ** use it in your train method **
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **

  def setSmoothing(self, bestk):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.bestk = bestk

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method.
    """

    self.features = trainingData[0].keys() # this could be useful for your code later...

    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    else:
        kgrid = [self.bestk]

    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)

  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
    """
    Trains the classifier by collecting counts over the training data, and
    stores the Laplace smoothed estimates so that they can be used to classify.
    Evaluate each value of k in kgrid to choose the smoothing parameter
    that gives the best accuracy on the held-out validationData.

    trainingData and validationData are lists of feature Counters.  The corresponding
    label lists contain the correct label for each datum.

    To get the list of all possible features or labels, use self.features and
    self.legalLabels.
    """

    "*** YOUR CODE HERE ***"

    self.trainingSize = len(trainingData)
    self.cdt = {}
    self.cpt = {}
    for legalLabel in self.legalLabels:
      self.cdt[legalLabel] = util.Counter()
      self.cpt[legalLabel] = {}

    for i in range(len(trainingData)):
      counter = trainingData[i]
      counterLabel = trainingLabels[i]
      self.cdt[counterLabel]["tCount"] += 1
      for feature in self.features:
        self.cdt[counterLabel][feature] += counter[feature]

    bestFeature = 0
    bestFeatureNum = 0
    for k in kgrid:
      for label in self.legalLabels:
        for feature in self.features:
          self.cpt[label][feature] = (self.cdt[label][feature] + k) / (0.0 + self.cdt[label]["tCount"] + 2*k)
    #Now check if this was a better k
      featureCorrect = 0
      for i in range(len(validationData)):
        possibleLabel = self.calculateLogJointProbabilities(validationData[i]).argMax()
        if(possibleLabel == validationLabels[i]):
          featureCorrect += 1
      if featureCorrect > bestFeatureNum:
        bestFeature = k
        bestFeatureNum = featureCorrect

      self.bestk = bestFeature
      for label in self.legalLabels:
        for feature in self.features:
          self.cpt[label][feature] = (self.cdt[label][feature] + self.bestk) / (0.0 + self.cdt[label]["tCount"] + 2*self.bestk)

  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.

    You shouldn't modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      posterior = self.calculateLogJointProbabilities(datum)
      guesses.append(posterior.argMax())
      self.posteriors.append(posterior)
    return guesses

  def calculateLogJointProbabilities(self, datum):
    """
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the log-joint counter, e.g.
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
    """
    logJoint = util.Counter()

    "*** YOUR CODE HERE ***"

    for label in self.legalLabels:
      lProb = math.log(self.cdt[label]["tCount"] / (self.trainingSize + 0.0))
      for feature in self.features:
        if(datum[feature]):
          lProb += math.log(self.cpt[label][feature])
        else:
          lProb += math.log(1 - self.cpt[label][feature])
      logJoint[label] = lProb

    return logJoint

  def findHighOddsFeatures(self, label1, label2):
        """
            Returns the 100 best features for the odds ratio:
                    P(feature=1 | label1)/P(feature=1 | label2)

            Note: you may find 'self.features' a useful way to loop through all possible features
            """
        oddFeatures = []
       
        o=util.Counter()

        "*** YOUR CODE HERE ***"
       
        for feature in self.features:
            o[feature]=self.cpt[label1][feature]/(0.0+self.cpt[label2][feature])
        oddFeatures=o.sortedKeys()
        del oddFeatures[100:]
        return oddFeatures