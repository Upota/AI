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
     Q(s,a) 중 가장 큰 액션을 취한다.
    """
    actions = gameState.getLegalActions(self.index)

    values = [self.evaluate(gameState, a) for a in actions]

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 1.0}

class OffensiveReflexAgent(ReflexCaptureAgent):
  def evaluate(self, gameState, action):
    """
    getFeatures 함수로부터 feature와 onReturn를 받는다.
    offensive Relflex Agent는 크게 두가지의 행동양식을 가지는데,
    첫번째는 상대방의 dot을 사냥하는 것이고,
    두번째는 일정량 이상의 dot을 채웠으면 최대한 자신의 진영으로 복귀하는데 집중하는 것이다.
    onReturn은 두번쨰 행동양식인지 아닌지를 알려준다. 0일 때는 사냥하는 행동양식, 1일 때 복귀하는 행동 양식이다.
    행동 양식에 따라 weight가 다르다.
    """
    features, onReturn = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action, onReturn)
    return features * weights

  def getFeatures(self, gameState, action):
    # 특징들을 계산하기에 앞서 필요한 변수들을 모아 놓았다.
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)    
    foodList = self.getFood(successor).asList()    
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
    capsules = self.getCapsules(gameState)

    teams = [successor.getAgentState(i) for i in self.getTeam(successor)]
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    ghosts = [a for a in enemies if (not a.isPacman) and a.getPosition() != None]

    onReturn = 0
    features['Score'] = -len(foodList)      # dot을 먹는 것에 집중하기 위해 기본 스코어를 남은 음식의 양으로 설정하였다.
    if action == Directions.STOP: features['stop'] = 1      # 멈춰있는 상태에 패널티를 부여
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1                       # 자신의 이전 행동으로 반복할 우여가 있어, 이에 대한 패널티를 적용하였다.

    features['onAttack'] = -1                      # 팩맨이 아닐때는 상대 진영으로 가야하므로, Attack모드가 되는 것을 선호하도록 한다.
    if myState.isPacman:
      features['onAttack'] = 1

    features['eatCapsule'] = 0              # 캡슐을 먹는 것을 선호한다.
    for c in capsules:
      if c == myPos:
        features['eatCapsule'] = 1
        break

    numCarrying = myState.numCarrying         # 현재 가지고 있는 음식의 양과 옮기는데 성공한 음식의 양 정보들이다.
    features['numCarrying'] = numCarrying
    numReturned = myState.numReturned
    features['numReturned'] = numReturned

    features['ghostISclose'] = 0            # 상대방 진영의 유령과의 거리를 계산하였다.
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
        elif mindist < 4:                          # 유령이 가까이 있음을 경고한다.
          features['ghostISclose'] = 4
      else: 
        if myPos == closestGhost.getPosition():   # 유령이 캡슐에 인해 겁에 질린 상태이면 오히려 잡아먹도록 한다.
          features['eatGhost'] = 1

    if gameState.getAgentState(self.index).numCarrying > 5:
      onReturn = 1
      mindist = min([self.getMazeDistance(myPos, a.getPosition()) for a in teams if a != myState])    # 자기 팀 진영으로 복귀하기위한 길잡이다.
      features['dist'] = mindist

    if onReturn:
      features['Score'] = self.getScore(gameState) - len(foodList)      # 복귀 상황에서는 dot을 먹는 것을 고려하지 않으므로 먼저 특징요소들을 반환한다.
      return features, onReturn

    if len(foodList) > 0:                                   # 자신의 위치에서 모든 음식들의 거리 정보와 음식의 가까운 유령과의 거리 정보를 복합하여 나아가야 할 길을 알려준다.
      Distances = []
      for food in foodList:
        Distance = self.getMazeDistance(myPos, food)
        if len(ghosts) > 0:
            mindist = min([self.getMazeDistance(food, a.getPosition()) for a in ghosts])
        else:
            mindist = 0
        Distances.append(Distance - (mindist * 0.5))
      features['distanceToFood'] = min(Distances)

    return features, onReturn

  def getWeights(self, gameState, action, onReturn):
    """
    행동양식에 따라 가중치가 다르다
    """
    if onReturn:
      return {'Score': 10, 'numReturned': 10, 'stop': -15, 'onAttack': -10, 'dist': -5, 'ghostISclose': -2, 'eatCapsule': 15, 'eatGhost': 10}

    return {'Score': 10, 'distanceToFood': -1, 'numCarrying': 2, 'numReturned': 10, 'stop': -15, 'onAttack': 10, 'ghostISclose': -2, 'eatCapsule': 15, 'eatGhost': 10, 'reverse': -5}

class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  우리팀 진영에 들어온 상대팀으로부터 dot을 방어하는 역할 맡는다.
  공격에는 관여하지 않는다.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    features['onDefense'] = 1             # 유령상태이면 방어상태이므로 방어를 선호하도록 한다.
    if myState.isPacman: features['onDefense'] = 0

    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    ghosts = [a for a in enemies if (not a.isPacman) and a.getPosition() != None]

    features['numInvaders'] = len(invaders)               # 침입자를 발견하면 즉시 방어하도록 한다.
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if len(ghosts) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in ghosts]      # 침입자가 없을 때, 상대방이 공격하기위해 유령상태에서 다가올 것이므로, 미리 해당 유령과 가까운 지점에서 진을 친다.
      features['nearestGhost'] = min(dists)
    
    if action == Directions.STOP: features['stop'] = 1              # 멈춰있는 액션과 반복의 가능성이 있는 액션들에 패널티를 부여하고, 시작지점에서 필드로 나아가기 위한 펌프역할을 한다.
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'nearestGhost': -1, 'stop': -100, 'reverse': -2}