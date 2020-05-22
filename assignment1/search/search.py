# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    
    pos = problem.getStartState()
    fringe = util.Stack()
    visited = util.Stack()
    visited.push(pos)
    fringe.push([(pos, 'Action', 'Cost')])
    find, ret_val = recursive_dfs(problem, fringe, visited)

    return ret_val

def recursive_dfs(problem, fringe, visited):
    ret_val = []
    path = fringe.pop()
    pos = path[-1]
    if problem.isGoalState(pos[0]):
        for node in path[1:]:
            ret_val.append(node[1])

        return True, ret_val

    for succ in problem.getSuccessors(pos[0]):
            if succ[0] in visited.list:
                continue
            
            nodes = path.copy()
            nodes.append(succ)
            fringe.push(nodes)
            visited.push(succ[0])
            find, ret = recursive_dfs(problem, fringe, visited)
            if find:
                return find, ret

    return False, ret_val

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    ret_val = []
    pos = problem.getStartState()
    fringe = util.Queue()
    visited = util.Stack()
    fringe.push([(pos,'Action', 'Cost')])
    visited.push(pos)

    while True:
        path = fringe.pop()
        pos = path[-1]
        if problem.isGoalState(pos[0]):
            for node in path[1:]:
                ret_val.append(node[1])
            return ret_val

        for succ in problem.getSuccessors(pos[0]):
            if succ[0] in visited.list:
                continue
            nodes = path.copy()
            nodes.append(succ)
            fringe.push(nodes)
            visited.push(succ[0])

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    ret_val = []
    pos = problem.getStartState()
    fringe = util.PriorityQueue()
    visited = util.Stack()
    path = {pos : [(pos, 'Action', 0)]}
    fringe.push(pos, 0)
    
    while True:
        state = fringe.pop()
        visited.push(state)
        
        if problem.isGoalState(state):
            for node in path[state]:
                ret_val.append(node[1])
            return ret_val[1:]

        for succ in problem.getSuccessors(state):
            if succ[0] in visited.list:
                continue
            nodes = path[state].copy()
            nodes.append(succ)
            cost = 0
            
            for node in nodes:
                cost += node[2]
            
            if path.get(succ[0]) != None:
                cumul_cost = 0
                for node in path[succ[0]]:
                    cumul_cost += node[2]
                
                if cumul_cost < cost:
                    continue
            
            path[succ[0]] = nodes
            fringe.update(succ[0], cost) 

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    ret_val = []
    pos = problem.getStartState()
    fringe = util.PriorityQueue()
    visited = util.Stack()
    path = {pos : [(pos, 'Action', 0)]}
    fringe.push(pos, 0)
    
    while True:
        state = fringe.pop()
        visited.push(state)
        
        if problem.isGoalState(state):
            for node in path[state]:
                ret_val.append(node[1])
            return ret_val[1:]

        for succ in problem.getSuccessors(state):
            if succ[0] in visited.list:
                continue
            nodes = path[state].copy()
            nodes.append(succ)
            cost = heuristic(succ[0], problem)

            for node in nodes:
                cost += node[2]
            
            if path.get(succ[0]) != None:
                cumul_cost = heuristic(succ[0], problem)

                for node in path[succ[0]]:
                    cumul_cost += node[2]

                if cumul_cost < cost:
                    continue

            path[succ[0]] = nodes
            fringe.update(succ[0], cost) 

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
