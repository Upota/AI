# alpha beta pruning

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

class AlphaBetaAgent(CaptureAgent):
  """
  alpha-beta 가지치기를 적용한 모델이다.
  feature를 얻는 것과 가중치는 reflex agent와 대부분 동일하다.
  """
 
  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    self.start = gameState.getAgentPosition(self.index)
    self.order = self.getOpponents(gameState)
    self.order.insert(0,self.index)
    self.depth = 2          # 탐색 깊이는 2로 설정하였다.
    self.bouns = 0          # 보너스 점수는 공격형 agent가 자신의 행동양식을 전환하는데에 중요한 역할을 한다.

  def chooseAction(self, gameState):

    start = time.time()
    _, action = self.value(gameState, 0, 0, None, None)
    t = time.time() - start

    return action
  
  def value(self, gameState, orderIdx, current_depth, alpha, beta):                 # alpha-beta 가지치기를 위한 세 함수 들이다. 
    if current_depth == self.depth:
      value = self.evaluate(gameState, None)                                       
      return value, None

    if orderIdx == 0:
      return self.max_layer(gameState, orderIdx, current_depth, alpha, beta)
  """
  세 함수들의 반환형은 val, act이다.
  value함수에서 계산한 리워드를 통해 max와 min 레이어는 자신들의 역할을 수행해내고, 가장 처음으로 호출 된 max에서는 최선의 액션을 반환해야하므로 이를 act에 담아 반환한다.
  chooseAction에서는 그대로 act를 반환하여 agent의 행동을 알려준다.
  """
  def max_layer(self, gameState, orderIdx, current_depth, alpha, beta):         # max를 골라내는 레이어이다.
    agent = self.order[orderIdx]
    LegalMoves = gameState.getLegalActions(agent)
    LegalMoves.remove('Stop')                       # 계산 공간을 줄이기위해 stop액션을 모두 제외하고 계산하였다.
    max_val = float('-inf')
    max_act = []
    a = alpha
    b = beta

    for action in LegalMoves:
      succ = gameState.generateSuccessor(agent, action)
      val, _ = self.min_layer(succ, orderIdx+1, current_depth, a, b)
      if a == None:                                                       # alpha가 비어있다면 현재 얻은 첫 value를 alpha로 설정한다.
        a = val
      if max_val == val:
        max_act.append(action)
      elif max_val < val:
        max_val = val
        max_act = [action]
      if b != None:                                   # beta가 비어있지 않을때 조건문이 실행되도록 한다.
        if b < max_val:
          return max_val, max_act                 # 여기서 바로 return 함으로써 가지치기가 수행된다.
      a = max(a, max_val)

    return max_val, random.choice(max_act)
    
  def min_layer(self, gameState, orderIdx, current_depth, alpha, beta):           # max_layer 함수와 역할이 반대일뿐 알고리즘은 온전히 동일하다
    agent = self.order[orderIdx]
    LegalMoves = gameState.getLegalActions(agent)
    LegalMoves.remove('Stop')
    min_val = float('inf')
    min_act = []
    a = alpha
    b = beta

    for action in LegalMoves:
      succ = gameState.generateSuccessor(agent, action)
      if orderIdx == 2:
        val, _ = self.value(succ, 0, current_depth+1, a, b)
      else:
        val, _ = self.min_layer(succ, orderIdx+1, current_depth, a, b)
      if b == None:
        b = val
      if min_val == val:
        min_act.append(action)
      elif min_val > val:
        min_val = val
        min_act = [action]
      if a != None:
        if a > min_val:
          return min_val, min_act
      b = min(b, min_val)

    return min_val, random.choice(min_act)

  def getSuccessor(self, gameState, action):
    gameState = gameState.generateSuccessor(self.index, action)
    pos = gameState.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      return gameState.generateSuccessor(self.index, action)
    else:
      return gameState

  def evaluate(self, gameState, action):
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    features = util.Counter()
    features['Score'] = self.getScore(gameState)
    return features

  def getWeights(self, gameState, action):
    return {'Score': 1.0}

class OffensiveReflexAgent(AlphaBetaAgent):
  """
  이 부분은 앞서 기술한 reflex agent들과 역할은 비슷하다.
  따라서 동일한 주석들이 존재한다.
  가장 큰 차이점은 기존 reflex에서는 getfeature함수 내부에 전달받은 액션을 이용해 예상되는 다음 게임 상태로부터 정보를 얻어 가치를 계산하는데,
  이 모델에서는 전달받는 gameState가 이미 미래 상황에 대한 정보를 가지고 있기에 gameState를 활용하여 가치를 계산하는 점에서 약간의 차이점이 존재한다.
  """

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
    foodList = self.getFood(gameState).asList()    
    myState = gameState.getAgentState(self.index)
    myPos = myState.getPosition()
    capsules = self.getCapsules(gameState)
    
    prev = self.getPreviousObservation()    # 이전 상태를 이용하여 계산하는 부분도 있다.

    teams = [gameState.getAgentState(i) for i in self.getTeam(gameState)]
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    ghosts = [a for a in enemies if (not a.isPacman) and a.getPosition() != None]

    onReturn = 0
    features['Score'] = -len(foodList)  # dot을 먹는 것에 집중하기 위해 기본 스코어를 남은 음식의 양으로 설정하였다.
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1                       # 반복행동에 패널티 부여

    features['onAttack'] = -1                      # 공격상태 선호
    if myState.isPacman:
      features['onAttack'] = 1

    features['eatCapsule'] = 0      # 캡슐을 먹는 것을 선호한다
    for c in capsules:
      if c == myPos:
        features['eatCapsule'] = 1
        break

    numCarrying = myState.numCarrying     # 현재 가지고 있는 음식의 양과 옮기는데 성공한 음식의 양 정보들이다.
    features['numCarrying'] = numCarrying
    numReturned = myState.numReturned
    features['numReturned'] = numReturned

    features['ghostISclose'] = 0        # 상대방 진영의 유령과의 거리를 계산하였다.
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
        if myPos == closestGhost.getPosition():   # 유령이 캡슐에 인해 겁에 질린 상태이면 오히려 잡아먹도록 한다.
          features['eatGhost'] = 5

    if gameState.getAgentState(self.index).numCarrying > 6:   # 복귀 행동양식으로 돌아가는 여부를 결정한다.
      onReturn = 1
      if prev.getAgentState(self.index).numCarrying == 6:     # 만약 이전의 상태까지는 공격 행동양식이었다면, 복귀 행동양식이 되는 것에대해 부가 가치를 부여한다. 그 이유는 보고서에 자세히 기술하도록 하겠다.
        self.bouns += 1
        features['bonus'] = self.bouns
      mindist = min([self.getMazeDistance(myPos, a.getPosition()) for a in teams if a != myState])    # 자기 팀 진영으로 복귀
      features['dist'] = mindist

    if prev != None:
      prevState = prev.getAgentState(self.index)
      if prevState.numCarrying > 0 and numReturned - prevState.numReturned == prevState.numCarrying:
        self.bouns += 1                         # 위의 코드와 마찬가지로 dot의 반환에 성공하였을 때, 행동양식이 다시 공격모드로 바뀌므로 그에 대한 부가 가치를 부여한다.
        features['bonus'] = self.bouns

    if onReturn:
      features['Score'] = self.getScore(gameState) - len(foodList)  # 복귀 상황에서는 dot을 먹는 것을 고려하지 않으므로 먼저 특징요소들을 반환한다.
      return features, onReturn

    if len(foodList) > 0:     # 자신의 위치에서 모든 음식들의 거리 정보와 음식의 가까운 유령과의 거리 정보를 복합하여 나아가야 할 길을 알려준다.
      distToDot = [self.getMazeDistance(myPos, food) for food in foodList]
      if len(ghosts) > 0:
        distDotnGhost = [min([self.getMazeDistance(a.getPosition(), food) for a in ghosts]) for food in foodList]
        weightedDist = np.array(distToDot) - 0.3 * np.array(distDotnGhost)
      else:
        weightedDist = np.array(distToDot)
        
      features['distanceToFood'] = np.min(weightedDist)

    return features, onReturn

  def getWeights(self, gameState, action, onReturn):
    """
    행동양식에 따라 가중치가 다르다
    """
    if onReturn:
      return {'Score': 10, 'onAttack': -20, 'eatCapsule': 15, 'numCarrying': 10, 'numReturned': 10, 'ghostISclose': -3, 'dist': -1.5, 'eatGhost': 10, 'bonus': 50}

    return {'Score': 10, 'reverse': -15, 'onAttack': 20, 'eatCapsule': 15, 'numCarrying': 10, 'numReturned': 10, 'ghostISclose': -3, 'eatGhost': 10, 'distanceToFood': -2, 'bonus': 50}

class DefensiveReflexAgent(AlphaBetaAgent):
  """
  우리팀 진영에 들어온 상대팀으로부터 dot을 방어하는 역할 맡는다.
  공격에는 관여하지 않는다.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    myState = gameState.getAgentState(self.index)
    myPos = myState.getPosition()

    features['onDefense'] = 1     # 유령상태이면 방어상태이므로 방어를 선호하도록 한다.
    if myState.isPacman: features['onDefense'] = 0

    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    ghosts = [a for a in enemies if (not a.isPacman) and a.getPosition() != None]

    features['numInvaders'] = len(invaders)     # 침입자를 발견하면 즉시 방어하도록 한다.
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if len(ghosts) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in ghosts]    # 침입자가 없을 때, 상대방이 공격하기위해 유령상태에서 다가올 것이므로, 미리 해당 유령과 가까운 지점에서 진을 친다.
      features['nearestGhost'] = min(dists)
    
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction] # 반복의 가능성이 있는 액션들에 패널티를 부여하고, 시작지점에서 필드로 나아가기 위한 펌프역할을 한다.
    if action == rev: features['reverse'] = 1                         # 'stop' 액션이 미리 제외되어있으므로 여기에서는 reflex agent와는 달리 stop에 관해 따로 더 다루지 않았다.

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'nearestGhost': -1, 'reverse': -2}