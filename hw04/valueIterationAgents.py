# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
  def __init__(self, mdp, discount=0.9, iterations=100):
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        allStates = self.mdp.getStates()        

        # Write value iteration code here

        for i in range(iterations):
          interState = util.Counter()
          for state in allStates:
            best = -9999999
            actions = mdp.getPossibleActions(state)
            for action in actions:
              transitions = self.mdp.getTransitionStatesAndProbs(state, action)
              sumTransitions = 0
              for transition in transitions:
                reward = self.mdp.getReward(state, action, transition[0])
                sumTransitions += transition[1]*(reward + discount*self.values[transition[0]])
              best = max(best, sumTransitions)
            if best != -9999999:
              interState[state] = best          
          
          for state in allStates:
            self.values[state] = interState[state]


    
  def getValue(self, state):
    return self.values[state]

  def getQValue(self, state, action):
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        sumTransitions = 0
        for transition in transitions:
          reward = self.mdp.getReward(state, action, transition[0])
          sumTransitions += transition[1]*(reward + self.discount*self.values[transition[0]])

        return sumTransitions

  def getPolicy(self, state):
        a = None
        currValues = util.Counter()
        actions = self.mdp.getPossibleActions(state)
        for action in actions:
          currValues[action] = self.getQValue(state, action)
        policy = currValues.argMax()
        return policy

  def getAction(self, state):
    return self.getPolicy(state)
  
