# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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
        remaining newFood (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost_idx will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghost_idx.scaredTimer for ghost_idx in newGhostStates]
        
        score = currentGameState.getScore()                     # 현재 점수를 기준으로 계산

        if successorGameState.isLose() or action == 'Stop':     # 게임을 패배나 Stop 액션을 방지
            score -= 100
            return score
        if successorGameState.isWin():                          # 게임 승리시 바로 큰 점수 반환
            score += 1000
            return score

        if currentGameState.hasFood(newPos[0], newPos[1]):      # 해당 액션으로 음식을 바로 먹을 수 있으면 큰 점수
            score += 400

        for capsule in currentGameState.getCapsules():            # 캡슐을 먹을 수 있으면 큰 점수
            if capsule == newPos:
                score += 400
                continue
            dist = manhattanDistance(newPos,capsule)                # 캡슐도 하나의 dot처럼 여겨 가까이 가도록 유도한다.
            discount_score = 50                                     # 거리가 캡슐에서 거리가 멀수록 얻는 점수가 적도록 discount를 적용하였다.
            for k in range(dist):
                discount_score *= 0.9
            score += discount_score                                 # 캡슐에 가까이 가는걸 선호

        for ghost in newGhostStates:                        # 유령을 피하는걸 선호
            ghostPos = ghost.getPosition()
            dist = manhattanDistance(ghostPos,newPos)
            score += dist * 0.5

        for x in range(-2,3):             # 주변에 음식이 있으면 플러스 점수. 2 걸음내에 도달 할 수 있는 지점을 탐색.
            for y in range(-2,3):
                handle = (newPos[0]+x,newPos[1]+y)      # handle은 탐색할 지점
                try:
                    has_food = newFood[handle[0]][handle[1]]    # dot이 해당 위치에 존재하는지 확인
                except IndexError:                              # 주어진 map의 크기를 몰라서 발생할 수 있는 인덱스 에러를 처리함.
                    has_food = False
                if has_food:
                    dist = manhattanDistance(newPos,handle)         # 할인률을 적용
                    discount_score = 50
                    for k in range(dist):
                        discount_score *= 0.9
                    score += discount_score
                                        
        discount_score = 50             # 음식이 있는 쪽으로 이동. 주변에 음식이 없을때 방향을 알려주기 위함.
        dot_list = newFood.asList()
        dist = manhattanDistance(newPos,dot_list[0])
        for k in range(dist):
            discount_score *= 0.9
        score += discount_score

        return score

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
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        val, action = self.value(gameState, 0, 0)       # 함수들의 반환형이 val, action의 꼴인데 val은 max, min을 계산하는 단계에서 필요하고,
                                                        # action은 결국 마지막에 어떤 액션을 취해야하는지 결정적으로 알려주기위해 필요했다.
        return action
    
    def value(self, State, next_agent, current_depth):      
        if State.isWin() or State.isLose() or current_depth == self.depth:  # leaf node를 처리. 해당 상태의 점수를 반환한다.
            return self.evaluationFunction(State), None
        if next_agent == 0:                                                 # next_agent는 다음 agent가 max인지 min인지 더불어 min이면 어떤 ghost인지를 구분해준다.
            return self.max_layer(State, current_depth)                     # next_agent가 0이면 max.
       
        return self.min_layer(State, next_agent, current_depth)

    def max_layer(self, State, current_depth):
        LegalMoves = State.getLegalActions(0)
        max_val = -999999
        max_act = LegalMoves[0]

        for action in LegalMoves:                                       # 가능한 액션들로만 탐색한다.
            succ = State.generateSuccessor(0, action)
            val, _ = self.value(succ, 1, current_depth)                 # 해당 액션을 시행했을때의 value를 얻고
            if max_val <  val:                                          # 가장 큰 것을 찾는다.
                max_val = val
                max_act = action

        return max_val, max_act
    
    def min_layer(self, State, agentIdx, current_depth):            # max_layer 함수와 역할이 반대일 뿐 알고리즘은 동일하다.
        LegalMoves = State.getLegalActions(agentIdx)
        min_val = 999999
        min_act = LegalMoves[0]

        for action in LegalMoves:
            succ = State.generateSuccessor(agentIdx, action)
            if State.getNumAgents()-1 == agentIdx:                  # 현재 agent가 마지막 agent인지 확인하고 마지막이면 max로 넘어감.
                val, _ = self.value(succ, 0, current_depth+1)
            else:
                val,_ = self.value(succ, agentIdx+1, current_depth)     # 현재 agent가 마지막 agent가 아니면 다음 agent를 탐색
            if min_val > val:
                min_val = val
                min_act = action

        return min_val, min_act

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        
        val, action = self.value(gameState, 0, 0, None, None)

        return action
    
    def value(self, State, next_agent, current_depth, alpha, beta):                 # minmax와 대부분 동일하다. 달라진 점은 alpha와 beta를 인자로 넘겨주고, 이들의 value를 결정하는 적절한 위치에 코드를 삽입했다.
        if State.isWin() or State.isLose() or current_depth == self.depth:
            return self.evaluationFunction(State), None
        if next_agent == 0:
            return self.max_layer(State, current_depth, alpha, beta)
       
        return self.min_layer(State, next_agent, current_depth, alpha, beta)

    def max_layer(self, State, current_depth, alpha, beta):
        LegalMoves = State.getLegalActions(0)
        max_val = -999999
        max_act = LegalMoves[0]
        a = alpha
        b = beta

        for action in LegalMoves:
            succ = State.generateSuccessor(0, action)
            val, _ = self.value(succ, 1, current_depth, a, b)
            if a == None:                                                       # alpha가 비어있다면 현재 얻은 첫 value를 alpha로 설정한다.
                a = val
            if max_val <  val:
                max_val = val
                max_act = action
            if b != None:                                   # beta가 비어있지 않을때 조건문이 실행되도록 한다.
                if b < max_val:
                    return max_val, max_act                 # 여기서 바로 return 함으로써 가지치기가 수행된다.
            a = max(a, max_val)

        return max_val, max_act
    
    def min_layer(self, State, agentIdx, current_depth, alpha, beta):           # max_layer 함수와 역할이 반대일뿐 알고리즘은 온전히 동일하다
        LegalMoves = State.getLegalActions(agentIdx)
        min_val = 999999
        min_act = LegalMoves[0]
        a = alpha
        b = beta

        for action in LegalMoves:
            succ = State.generateSuccessor(agentIdx, action)
            if State.getNumAgents()-1 == agentIdx:
                val, _ = self.value(succ, 0, current_depth+1, a, b)
            else:
                val, _ = self.value(succ, agentIdx+1, current_depth, a, b)
            if b == None:
                b = val
            if min_val > val:
                min_val = val
                min_act = action
            if a != None:
                if a > min_val:
                    return min_val, min_act
            b = min(b, min_val)

        return min_val, min_act

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
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost_idx-hunting, pellet-nabbing, newFood-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
