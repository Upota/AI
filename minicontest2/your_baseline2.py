# reflex agent another agent form
# agents acting version can be change

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
               first = 'ReflexCaptureAgent', second = 'ReflexCaptureAgent'):
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
  """
  세 가지 행동양식이 존재한다.
  agent는 자신의 행동양식이 무엇인지 act_ver를 통해 알아낸다.
  0. 방어
  1. dot사냥
  2. dot사냥 후 복귀
  """
  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    self.start = gameState.getAgentPosition(self.index)
    self.T = self.getTeam(gameState)
    self.teamIdx = [t for t in self.T if t != self.index]
    self.act_ver = {}           # agent들의 현재 행동양식 정보를 저장한다. 첫번째 agent는 공격이고 두번째는 방어가 할당되는 과정이다.
    ver = 1                             
    for t in self.T:
      self.act_ver[t] = ver
      if ver == 0:
        ver = 1
      else:
        ver = 0

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
    successor = self.getSuccessor(gameState, action)        # 계산에 필요한 여러 변수들을 미리 선언하였다.
    foodList = self.getFood(successor).asList()    
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
    capsules = self.getCapsules(gameState)

    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    ghosts = [a for a in enemies if (not a.isPacman) and a.getPosition() != None]

    curr_enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    curr_invaders = [a for a in curr_enemies if a.isPacman and a.getPosition() != None]
    # curr_ghosts = [a for a in curr_enemies if (not a.isPacman) and a.getPosition() != None]
    
    if len(curr_invaders) == 0:                           # 현재 침입자가 없을 때, 기존의 공격요원이 죽었다면, 방어하고 있던 요원이 새로 부활한 요원보다 상대방의 dot에 가까이 있을 것이므로 공수 교대한다.
      if self.act_ver[self.index] == 0:      # defensive
        for t in self.teamIdx:                            # 공수 교대하는 일련의 과정들이다.
          if self.getMazeDistance(gameState.getAgentPosition(t), self.start) < 3 and self.getMazeDistance(gameState.getAgentPosition(self.index),self.start) > 3:
            self.act_ver[self.index] = 1
            self.act_ver[t] = 0
      elif (self.act_ver[self.index] == 1 or self.act_ver[self.index] == 2):  # offensive
        for t in self.teamIdx:
          if self.getMazeDistance(gameState.getAgentPosition(t), self.start) > 3 and self.getMazeDistance(gameState.getAgentPosition(self.index),self.start) < 3:
            self.act_ver[self.index] = 0
            self.act_ver[t] = 1
          
    for t in self.teamIdx:                              # 두 요원이 동일한 행동양식을 가지는 예상치 못한 상황에 대비해 새로 행동양식을 부과한다.
      if self.act_ver[t] == self.act_ver[self.index]:
        ver = 1
        for a in self.T:
          self.act_ver[a] = ver
          if ver == 1:
            ver = 0
          else:
            ver = 1

    if self.act_ver[self.index] == 0:      # defensive      방어 행동양식을 따를 때, 얻는 특징들이다.
      features['onDefense'] = 1             # 유령상태이면 방어상태이므로 방어를 선호하도록 한다.
      if myState.isPacman: features['onDefense'] = 0

      features['numInvaders'] = len(invaders)   # 침입자를 발견하면 즉시 방어하도록 한다.
      if len(invaders) > 0:
        dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
        features['invaderDistance'] = min(dists)

      if len(ghosts) > 0:
        dists = [self.getMazeDistance(myPos, a.getPosition()) for a in ghosts]    #  침입자가 없을 때, 상대방이 공격하기위해 유령상태에서 다가올 것이므로, 미리 해당 유령과 가까운 지점에서 진을 친다.
        features['nearestGhost'] = min(dists)
    
      if action == Directions.STOP: features['stop'] = 1
      rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]   # 멈춰있는 액션과 반복의 가능성이 있는 액션들에 패널티를 부여하고, 시작지점에서 필드로 나아가기 위한 펌프역할을 한다.
      if action == rev: features['reverse'] = 1


    if self.act_ver[self.index] == 1 or self.act_ver[self.index] == 2:        # offensive
      if len(foodList) > 0:
        Distances = []
        for food in foodList:
          Distance = self.getMazeDistance(myPos, food)
          if len(ghosts) > 0:
            mindist = min([self.getMazeDistance(food, a.getPosition()) for a in ghosts])
          else:
            mindist = 0
        Distances.append(Distance - (mindist * 0.5))

      features['distanceToFood'] = min(Distances)      # 자신의 위치에서 모든 음식들의 거리 정보와 음식의 가까운 유령과의 거리 정보를 복합하여 나아가야 할 길을 알려준다.

      features['Score'] = -len(foodList)
      if action == Directions.STOP: features['stop'] = 1      # 멈춰있는 상태에 패널티를 부여
      rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
      if action == rev: features['reverse'] = 1                       # 자신의 이전 행동으로 반복할 우여가 있어, 이에 대한 패널티를 적용하였다.

      features['eatCapsule'] = 0      # 캡슐을 먹는 것을 선호한다.
      for c in capsules:
        if c == myPos:
          features['eatCapsule'] = 1
          break

      numCarrying = myState.numCarrying
      features['numCarrying'] = numCarrying
      numReturned = myState.numReturned
      features['numReturned'] = numReturned

      features['onAttack'] = -1                      # 팩맨이 아닐때는 상대 진영으로 가야하므로, Attack모드가 되는 것을 선호하도록 한다.
      if myState.isPacman:
        features['onAttack'] = 1
        if len(ghosts) > 0:
          dists = [self.getMazeDistance(myPos, a.getPosition()) for a in ghosts]
          mindist = min(dists)
        #   features['ghostDistance'] = mindist
          for a, v in zip(ghosts, dists):
            if v == mindist:
              closestGhost = a
          if closestGhost.scaredTimer == 0:
            if mindist == 1:                           # 유령이 너무 가까우면 도망친다.
              features['ghostISclose'] = 7
              self.act_ver[self.index] = 2
            elif mindist < 4:                          # 유령이 가까이 있음을 경고
              features['ghostISclose'] = 1
          else:
            if myPos == closestGhost.getPosition():
              features['eatGhost'] = 1          # 유령이 캡슐에 인해 겁에 질린 상태이면 오히려 잡아먹도록 한다.
        else:
          features['ghostISclose'] = 0

      if gameState.getAgentState(self.index).numCarrying > 5:     # 일정량 이상 dot을 먹었으면 복귀모드로 전환한다.
        self.act_ver[self.index] = 2

      if not myState.isPacman:        # return -> offensive     만약 복귀모드였다면 유령이 되었을 시, 복귀모드는 무조건 종료하게 된다.
        self.act_ver[self.index] = 1

    if self.act_ver[self.index] == 2:      # return Carrying Dots
      mindist = min([self.getMazeDistance(myPos, successor.getAgentPosition(a)) for a in self.teamIdx])    # 자기 팀 진영으로 복귀
      features['dist'] = mindist
      features['Score'] = self.getScore(gameState) - len(foodList)

    return features

  def getWeights(self, gameState, action):
    ver = self.act_ver[self.index]      # 행동 양식에 따라 반환하는 weight가 다르다.
    if ver == 0:
      return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'nearestGhost': -1, 'stop': -100, 'reverse': -2}

    if ver == 1:        # offensive
      return {'Score': 10, 'distanceToFood': -1, 'numCarrying': 2, 'numReturned': 10, 'stop': -15, 'onAttack': 10, 'ghostISclose': -2, 'eatCapsule': 15, 'eatGhost': 10, 'reverse': -5}
    
    if ver == 2:        # return 
      return {'Score': 10, 'numReturned': 10, 'stop': -15, 'onAttack': -10, 'dist': -5, 'ghostISclose': -2, 'eatCapsule': 15, 'eatGhost': 10}