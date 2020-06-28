# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math
from collections import defaultdict
import pickle
import numpy as np

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        "*** YOUR CODE HERE ***"
        self.qValues = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        key = (state,action)
        v = self.qValues[key]
        
        return v

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)legalActions
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        legalActions = self.getLegalActions(state)
        if (legalActions == () or legalActions == []):
          return 0

        max_v = float('-inf')

        for a in legalActions:
          v = self.getQValue(state,a)
          if v > max_v:
            max_v = v
      
        return max_v

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        legalActions = self.getLegalActions(state)
        if (legalActions == () or legalActions == []):
          return None

        best_a = legalActions[0]
        max_v = float('-inf')

        for a in legalActions:
          v = self.getQValue(state,a)
          if v >= max_v:
            max_v = v
            best_a = a

        return best_a

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        if (flipCoin(self.epsilon)):
          return random.choice(legalActions)

        action = self.computeActionFromQValues(state)

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe aq6
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        new_v = (1 - self.alpha) * (self.getQValue(state,action)) + self.alpha * (reward + self.discount * (self.getValue(nextState)))
        self.qValues[(state,action)] = new_v

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()
        self.cum_weights = defaultdict(list)
        self.avgQV = defaultdict(list)
        self.qv = util.Counter()
        self.alpha = 1.0
        # self.epsilon = 1.0
        # self.eidx = "epsilon=%.1f" %self.epsilon       # code for discuss #3
        self.idx = "alpha=%.1f" %self.alpha       # code for discuss #4

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        qv = 0
        feats = self.featExtractor.getFeatures(state,action)
        keys = feats.keys()

        for k in keys:
          qv += self.weights[k] * feats[k]

        return qv

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        max_v = self.computeValueFromQValues(nextState)
        diff = (reward + self.discount * max_v) - self.getQValue(state,action)

        feats = self.featExtractor.getFeatures(state,action)
        keys = feats.keys()

        for k in keys:
          self.weights[k] += self.alpha * diff * feats[k]

        self.qv[(state,action)] = self.getQValue(state,action)
        "***  DO NOT DELETE BELOW ***"
        self.write()

    def write(self):
        """
          DO NOT DELETE
        """
        for i in ["bias", "#-of-ghosts-1-step-away", "eats-food", "closest-food"]:
            self.cum_weights[i].append(self.weights[i])

    def save(self):
        """
          DO NOT DELETE
        """
        with open('./cmu_weights.pkl','wb') as f:
            pickle.dump(self.cum_weights,f)

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)
        
        trainAvg = (self.qv.totalCount() / len(self.qv))

        self.avgQV[self.idx].append(trainAvg)
        fname = './alpha/' + self.idx + '.pkl'
        with open(fname,'wb') as f:
            pickle.dump(self.avgQV,f)
        
        # self.avgQV[self.eidx].append(trainAvg)
        # fname = './epsilon/' + self.eidx + '.pkl'
        # with open(fname,'wb') as f:
        #     pickle.dump(self.avgQV,f)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            "***  DO NOT DELETE BELOW ***"
            self.save()