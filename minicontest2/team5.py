# expecti-max update
# modify value and getFeatures

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

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class ExpectiMaxAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
 
  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    self.start = gameState.getAgentPosition(self.index)
    self.order = self.getOpponents(gameState)
    self.order.insert(0,self.index)
    self.depth = 2
    self.bouns = 0
    self.onret = 0

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """

    start = time.time()
    _, action = self.value(gameState, 0, 0)
    t = time.time() - start
    if t >= 1.0:
        util.pause()
    # print ('eval time for agent %d: %.4f' % (self.index, t))

    return action
  
  def value(self, gameState, orderIdx, current_depth):                 # compute
    if current_depth == self.depth:
      value = self.evaluate(gameState, None)
      return value, None

    if orderIdx == 0:
      return self.max_layer(gameState, orderIdx, current_depth)

  def max_layer(self, gameState, orderIdx, current_depth):
    agent = self.order[orderIdx]
    LegalMoves = gameState.getLegalActions(agent)
    LegalMoves.remove('Stop')
    max_val = -999999
    max_act = []

    for action in LegalMoves:
      succ = gameState.generateSuccessor(agent, action)
      val, _ = self.exp_layer(succ, orderIdx+1, current_depth)
      if max_val == val:
        max_act.append(action)
      elif max_val < val:
        max_val = val
        max_act = [action]

    return max_val, random.choice(max_act)
    
  def exp_layer(self, gameState, orderIdx, current_depth):          
    agent = self.order[orderIdx]
    LegalMoves = gameState.getLegalActions(agent)
    LegalMoves.remove('Stop')
    p = 1/(len(LegalMoves))
    exp_val = 0
    exp_act = []

    for action in LegalMoves:
      succ = gameState.generateSuccessor(agent, action)
      if orderIdx == 2:
        val, _ = self.value(succ, 0, current_depth+1)
      else:
        val, _ = self.exp_layer(succ, orderIdx+1, current_depth)
      exp_val += p * val

    return exp_val, exp_act

  def getSuccessor(self, gameState, action):
    """
    Finds the next gameState which is a grid position (location tuple).
    """
    gameState = gameState.generateSuccessor(self.index, action)
    pos = gameState.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      return gameState.generateSuccessor(self.index, action)
    else:
      return gameState

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
    features['Score'] = self.getScore(gameState)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'Score': 1.0}

class OffensiveReflexAgent(ExpectiMaxAgent):
  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features, onReturn = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action, onReturn)
    return features * weights

  def getFeatures(self, gameState, action):
    features = util.Counter()
    foodList = self.getFood(gameState).asList()    
    myState = gameState.getAgentState(self.index)
    myPos = myState.getPosition()
    capsules = self.getCapsules(gameState)
    
    prev = self.getPreviousObservation()

    teams = [gameState.getAgentState(i) for i in self.getTeam(gameState)]
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    ghosts = [a for a in enemies if (not a.isPacman) and a.getPosition() != None]

    onReturn = 0
    features['Score'] = -len(foodList)
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
          features['ghostISclose'] = 8
          onReturn = 1
        elif mindist < 4:                          # 유령이 가까이 있음을 경고
          features['ghostISclose'] = 3
      else:
        if myPos == closestGhost.getPosition():
          features['eatGhost'] = 5

    if gameState.getAgentState(self.index).numCarrying > 6:
      onReturn = 1
      if prev.getAgentState(self.index).numCarrying == 6:
        self.bouns += 1
        features['bonus'] = self.bouns
      mindist = min([self.getMazeDistance(myPos, a.getPosition()) for a in teams if a != myState])    # 자기 팀 진영으로 복귀
      features['dist'] = mindist

    if prev != None:
      prevState = prev.getAgentState(self.index)
      if prevState.numCarrying > 0 and numReturned - prevState.numReturned == prevState.numCarrying:
        self.bouns += 1
        features['bonus'] = self.bouns

    if onReturn:
      features['Score'] = self.getScore(gameState) - len(foodList)
      return features, onReturn

    # Compute distance to the nearest food with considering the nearest ghost

    if len(foodList) > 0:
      distToDot = [self.getMazeDistance(myPos, food) for food in foodList]
      if len(ghosts) > 0:
        distDotnGhost = [min([self.getMazeDistance(a.getPosition(), food) for a in ghosts]) for food in foodList]
        weightedDist = np.array(distToDot) - 0.3 * np.array(distDotnGhost)
      else:
        weightedDist = np.array(distToDot)
        
    #   dist1dots = [dist for dist in distToDot if dist == 1]
    #   dist2dots = [dist for dist in distToDot if dist == 2]
    #   discount = 0.8
    #   features['discountedDots'] = len(dist1dots) * discount + len(dist2dots) * discount * discount

      features['distanceToFood'] = np.min(weightedDist)

    # if len(invaders) > 0:
    #   dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
    #   features['invaderDistance'] = min(dists)

    return features, onReturn

  def getWeights(self, gameState, action, onReturn):
    if onReturn:
      return {'Score': 10, 'onAttack': -20, 'eatCapsule': 15, 'numCarrying': 10, 'numReturned': 10, 'ghostISclose': -3, 'dist': -1.5, 'eatGhost': 10, 'bonus': 50}

    return {'Score': 10, 'reverse': -15, 'onAttack': 20, 'eatCapsule': 15, 'numCarrying': 10, 'numReturned': 10, 'ghostISclose': -3, 'eatGhost': 10, 'distanceToFood': -2, 'bonus': 50}

class DefensiveReflexAgent(ExpectiMaxAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    myState = gameState.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
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