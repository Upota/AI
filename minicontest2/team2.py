# reflex agent upgrade

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint
import numpy as np

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
 
  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    self.start = gameState.getAgentPosition(self.index)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class OffensiveReflexAgent(ReflexCaptureAgent):
  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features, onReturn = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action, onReturn)
    return features * weights

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()    
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
    capsules = self.getCapsules(gameState)

    teams = [successor.getAgentState(i) for i in self.getTeam(successor)]
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    ghosts = [a for a in enemies if (not a.isPacman) and a.getPosition() != None]

    # curr_enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    # curr_ghosts = [a for a in enemies if (not a.isPacman) and a.getPosition() != None]

    onReturn = 0
    features['Score'] = -len(foodList)
    if action == Directions.STOP: features['stop'] = 1      # 멈춰있는 상태에 패널티를 부여
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1                       # 반복행동에 패널티 부여

    features['onAttack'] = -1                      # 공격상태 선호
    if myState.isPacman:
      features['onAttack'] = 1

    features['eatCapsule'] = 0
    for c in capsules:
      if c == myPos:
        features['eatCapsule'] = 1
        break

    numCarrying = myState.numCarrying
    features['numCarrying'] = numCarrying
    numReturned = myState.numReturned
    features['numReturned'] = numReturned

    features['ghostISclose'] = 0
    if len(ghosts) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in ghosts]
      mindist = min(dists)
      features['ghostDistance'] = mindist
      for a, v in zip(ghosts, dists):
        if v == mindist:
          closestGhost = a
      if closestGhost.scaredTimer == 0:
        if mindist == 1:                           # 유령이 너무 가까우면 도망친다.
          features['ghostISclose'] = 7
          onReturn = 1
        elif mindist < 4:                          # 유령이 가까이 있음을 경고
          features['ghostISclose'] = 4
      else:
        if myPos == closestGhost.getPosition():
          features['eatGhost'] = 1

    if gameState.getAgentState(self.index).numCarrying > 5:
      onReturn = 1
      mindist = min([self.getMazeDistance(myPos, a.getPosition()) for a in teams if a != myState])    # 자기 팀 진영으로 복귀
      features['dist'] = mindist

    if onReturn:
      features['Score'] = self.getScore(gameState) - len(foodList)
      return features, onReturn

    # Compute distance to the nearest food

    if len(foodList) > 0:
      Distances = []
      for food in foodList:
        Distance = self.getMazeDistance(myPos, food)
        if len(ghosts) > 0:
            mindist = min([self.getMazeDistance(food, a.getPosition()) for a in ghosts])
        else:
            mindist = 0
        Distances.append(Distance - (mindist * 0.5))
      features['distanceToFood'] = min(Distances)

    # if len(foodList) > 0: # This should always be True,  but better safe than sorry
    #   myPos = myState.getPosition()
    #   minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
    #   features['distanceToFood'] = minDistance

    return features, onReturn

  def getWeights(self, gameState, action, onReturn):
    if onReturn:
      return {'Score': 10, 'numReturned': 10, 'stop': -15, 'onAttack': -10, 'dist': -5, 'ghostISclose': -2, 'eatCapsule': 15, 'eatGhost': 10}

    return {'Score': 10, 'distanceToFood': -1, 'numCarrying': 2, 'numReturned': 10, 'stop': -15, 'onAttack': 10, 'ghostISclose': -2, 'eatCapsule': 15, 'eatGhost': 10, 'reverse': -5}

class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    ghosts = [a for a in enemies if (not a.isPacman) and a.getPosition() != None]

    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if len(ghosts) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in ghosts]
      features['nearestGhost'] = min(dists)
    
    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'nearestGhost': -1, 'stop': -100, 'reverse': -2}