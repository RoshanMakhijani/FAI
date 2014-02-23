from util import manhattanDistance
from game import Directions
import random, util
import math
from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  
    
  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.
    
    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best
    
    "Add more of your code here if you want to"
    
    return legalMoves[chosenIndex]
  
  def evaluationFunction(self, currentGameState, action):
    """
    Design a better evaluation function here. 
    
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.
    
    The code below extracts some useful information from the state, like the 
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    
    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generateSuccessor(0, action)
    nextPosition = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates() 
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    
    "*** YOUR CODE HERE ***"

    successorGameState = currentGameState.generateSuccessor(0, action)

    pacmanPosition = nextPos = successorGameState.getPacmanPosition()
    nextFood = successorGameState.getFood()
    foodList = nextFood.asList()
    bonus = 0
    if len(currentGameState.getFood().asList()) > len(foodList):
        bonus =10
    if len(foodList) == 0:
        return 100
    nextGhostStates = successorGameState.getGhostStates()
    score = 0
    minDistanceFood = 9999999
    for food in foodList:
        minDistanceFood = min(minDistanceFood, manhattanDistance(nextPosition,food))
    reciprocalFoodDis = float(1)/minDistanceFood
    score += reciprocalFoodDis
    minDisToGhost = 9999999
    for ghostState in nextGhostStates:
      minDisToGhost = min(minDisToGhost, manhattanDistance(nextPos,ghostState.getPosition()))
    #print 'dtg',distanceToGhost
    if minDisToGhost < 3:
        if minDisToGhost == 0:
            return -100
        else:
            score += minDisToGhost
    finalScore = score + bonus
    return finalScore

def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.
    
    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.
    
    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.
    
    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.  
  """
  
  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.isInitialStateProcessed = False
    self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """
  def maxValue(self, gameState, currdepth):
        scores = []
        numghosts = gameState.getNumAgents() - 1
        if gameState.isWin() or gameState.isLose() or self.depth == currdepth:
            return self.evaluationFunction(gameState)
        legalActions = gameState.getLegalActions(0)
        #legalActions.remove(Directions.STOP)
        i=0
        j=0
        resultingGameStates = []
        for action in legalActions:
            resultingGameStates.append(gameState.generateSuccessor(0, action))
            i=i+1
        for resultingGameState in resultingGameStates:                           
            scores.append(self.miniValue(resultingGameState, currdepth, 1))
            j=j+1
        return max(scores)
    
  def miniValue(self, gameState, currdepth, currGhostIndex):
        "numghosts = len(gameState.getGhostStates())"
        scores = []
        numghosts = gameState.getNumAgents() - 1
        if gameState.isWin() or gameState.isLose() or currdepth == self.depth:
            return self.evaluationFunction(gameState)
        legalActions = gameState.getLegalActions(currGhostIndex)
        i=0
        j=0
        resultingGameStates = []
        for action in legalActions:
            resultingGameStates.append(gameState.generateSuccessor(currGhostIndex, action))
            i=i+1
        if currGhostIndex == numghosts:
            for resultingGameState in resultingGameStates:
                scores.append(self.maxValue(resultingGameState, currdepth + 1))
                j=j+1
        else:
            for action in legalActions:
                scores.append(self.miniValue(gameState.generateSuccessor(currGhostIndex, action), currdepth, currGhostIndex+1))
                j=j+1
        return min(scores)
  
  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth 
      and self.evaluationFunction.
      
      Here are some method calls that might be useful when implementing minimax.
      
      gameState.getLegalActions(agentIndex):  
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1
      
      Directions.STOP:
        The stop direction, which is always legal
      
      gameState.generateSuccessor(agentIndex, action): 
        Returns the successor game state after an agent takes an action
      
      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    legalActions = gameState.getLegalActions(0)
    #legalActions.remove(Directions.STOP)
    i=0
    j=0
    scores = []
    resultingGameStates = []
    for action in legalActions:
        resultingGameStates.append(gameState.generateSuccessor(0, action))
        i=i+1
    for resultingGameState in resultingGameStates:       
        scores.append(self.miniValue(resultingGameState, 0, 1))
    bestScore = max(scores)
    bestIndices = []
    for index in range(len(scores)):
        if scores[index] == bestScore:
            bestIndices.append(index)
    chosenIndex = random.choice(bestIndices)
    if (not self.isInitialStateProcessed):
      self.initialStateProcessed = True
    return legalActions[chosenIndex]
    
class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """
  def maxValue(self, gameState, currdepth, alpha, beta):
        scores = []
        numghosts = gameState.getNumAgents() - 1
        if gameState.isWin() or gameState.isLose() or self.depth == currdepth:
            return self.evaluationFunction(gameState)
        legalActions = gameState.getLegalActions(0)
        #legalActions.remove(Directions.STOP)
        i=0
        j=0
        score = -9999999
        resultingGameStates = []
        for action in legalActions:
            resultingGameState= gameState.generateSuccessor(0, action)
            score = max(score, self.miniValue(resultingGameState, currdepth, 1, alpha, beta))
            if beta < score:
               return score
            alpha = max(alpha, score)
        return score
    
  def miniValue(self, gameState, currdepth, currGhostIndex, alpha, beta):
        "numghosts = len(gameState.getGhostStates())"
        scores = []
        numghosts = gameState.getNumAgents() - 1
        if gameState.isWin() or gameState.isLose() or currdepth == self.depth:
            return self.evaluationFunction(gameState)
        legalActions = gameState.getLegalActions(currGhostIndex)
        score = 9999999
        i=0
        j=0
        resultingGameStates = []
        for action in legalActions:
            resultingGameState = gameState.generateSuccessor(currGhostIndex, action)
            i=i+1
            if currGhostIndex == numghosts:
                score = min(score, self.maxValue(resultingGameState, currdepth + 1, alpha, beta))
            else:
                score = min(score, self.miniValue(gameState.generateSuccessor(currGhostIndex, action), currdepth, currGhostIndex+1, alpha, beta))
            if score < alpha:
                return score  
            beta = min(score, beta)  
        return score
   
  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    legalActions = gameState.getLegalActions(0)
    #legalActions.remove(Directions.STOP)
    bestAction = legalActions[0]
    alpha = -9999999
    beta = 9999999
    score = -9999999
    i=0
    j=0
    for action in legalActions:
        resultingGameState = gameState.generateSuccessor(0, action)
        resultingScore = self.miniValue(resultingGameState, 0, 1, alpha, beta)
        if resultingScore > score:
            score = resultingScore
            bestAction = action
        if resultingScore > beta:
            return bestAction
        alpha = max(alpha, resultingScore)
    if (not self.isInitialStateProcessed):
      self.initialStateProcessed = True
    return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
Your expectimax agent (question 4)
"""

  def getAction(self, gameState):
      """
Returns the expectimax action using self.depth and self.evaluationFunction

All ghosts should be modeled as choosing uniformly at random from their
legal moves.
"""
      "*** YOUR CODE HERE ***"
      return self.expectimax(gameState, 1, 0)

  def expectimax(self, gameState, currDepth, agentIndex):#pacman wants to maximize the score while the ghosts are minimzing
    if (currDepth > self.depth or gameState.isWin() or gameState.isLose()):
        return float(self.evaluationFunction(gameState))
    #get legal actions
    legalActions = []
    legalActions = gameState.getLegalActions(agentIndex)

    #compute parameters for next call
    nextAgentIndex = agentIndex + 1
    nextDepth = currDepth
    if nextAgentIndex >= gameState.getNumAgents():
            nextAgentIndex = 0
            nextDepth += 1

    nextScores = []
    for action in legalActions:
        nextScores.append(self.expectimax(gameState.generateSuccessor(agentIndex, action), nextDepth, nextAgentIndex))

    if agentIndex == 0 and currDepth == 1:
        bestIndex = nextScores.index(max(nextScores))
        return legalActions[bestIndex]
    elif agentIndex == 0:
        return max(nextScores)
    else:
        return float(sum(nextScores) / len(nextScores))  
      
def betterEvaluationFunction(currentGameState):
     """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).
    
      DESCRIPTION: <write something here so we know what you did>
     """
     "*** YOUR CODE HERE ***"
     nextPos = currentGameState.getPacmanPosition()
     nextFood = currentGameState.getFood()
     nextGhostStates = currentGameState.getGhostStates()
     nextScaredTimes = []
     ghostDistances = []
     for ghostState in nextGhostStates:
       nextScaredTimes.append(ghostState.scaredTimer)
       ghostDistances.append(manhattanDistance(ghostState.getPosition(), nextPos))
     cost = 0
     for scaredTime in nextScaredTimes:
         cost += scaredTime    
     foodList = nextFood.asList()
     wallList = currentGameState.getWalls().asList()
     foodNeighboursEmpty = 0
     foodDistances = []
     
     def getFoodNeighbors(foodPos):
         totFoodNeighbors = []
         totFoodNeighbors.append((foodPos[0]-1,foodPos[1]))
         totFoodNeighbors.append((foodPos[0],foodPos[1]-1))
         totFoodNeighbors.append((foodPos[0],foodPos[1]+1))
         totFoodNeighbors.append((foodPos[0]+1,foodPos[1]))
         return totFoodNeighbors
     i=0
     for food in foodList:
         neighbors = getFoodNeighbors(food)
         for neighbor in neighbors:
             if neighbor not in wallList and neighbor not in foodList:
                 foodNeighboursEmpty += 1
         foodDistances.append(manhattanDistance(nextPos, food))
    
     reciprocalFoodDistance = 0
     if len(foodDistances) > 0:
         reciprocalFoodDistance = (float)(1)/(min(foodDistances))
         cost += (float)(70)/len(foodDistances)
     else:
         cost+=1000
     cost += (2*(min(ghostDistances)*((reciprocalFoodDistance**8))) + currentGameState.getScore()-(float(foodNeighboursEmpty)*1.5))         
     return cost
  

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """
    
  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.
      
      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

