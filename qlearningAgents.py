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
#
# This code has been modified and extended for CS 1820 at Harvard University, 
# with adjustments tailored to align with the course curriculum and objectives.

from helpers.game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
from helpers.backend import ReplayMemory

import helpers.backend as backend

import random,util,math
import numpy as np
import copy

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
        "Initialize Q-values"
        ReinforcementAgent.__init__(self, **args)

        self.values = {}

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        # return the Q-value if exists, return 0.0
        if (state, action) in self.values:
            return self.values[(state, action)]
        return 0.0

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        # get legal actions for state
        legalActions = self.getLegalActions(state)
        
        # if are no legal actions, return 0.0
        if not legalActions:
            return 0.0
        
        # return maximum Q-value for all legal actions
        return max([self.getQValue(state, action) for action in legalActions])

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        # get legal actions for state
        legalActions = self.getLegalActions(state)
        
        # again if no legal actions return None
        if not legalActions:
            return None
        
        # find action with maximum Q-value
        # random.choice to break ties
        maxQValue = self.computeValueFromQValues(state)
        bestActions = [action for action in legalActions if self.getQValue(state, action) == maxQValue]
        return random.choice(bestActions)

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
        
        if not legalActions:
            return None
        
        # with probability epsilon, choose a random action (exploration)
        # oherwise, choose the best action according to Q-values (exploitation)
        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        else:
            action = self.computeActionFromQValues(state)
            
        return action

    def update(self, state, action, nextState, reward: float):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
          NOTE: You should never call this function,
          it will be called on your behalf
        """
        # Q-learning update rule: Q(s,a) = Q(s,a) + alpha * [r + gamma * max_a'(Q(s',a')) - Q(s,a)]
        # calc the sample: reward + discount * max_a'(Q(s',a'))
        sample = reward + self.discount * self.computeValueFromQValues(nextState)
        
        # upldate Q-value for (state, action) with learning rate (alpha)
        currentQValue = self.getQValue(state, action)
        self.values[(state, action)] = currentQValue + self.alpha * (sample - currentQValue)

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

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        # get features for state-action pair
        features = self.featExtractor.getFeatures(state, action)
        
        # Q(s,a) = w * feature vector (dot product)
        qValue = 0.0
        for feature, value in features.items():
            qValue += self.weights[feature] * value
            
        return qValue

    def update(self, state, action, nextState, reward: float):
        """
           Should update your weights based on transition
        """
        features = self.featExtractor.getFeatures(state, action)
        
        # r + γ * max_a'(Q(s',a')) - Q(s,a)
        difference = (reward + self.discount * self.computeValueFromQValues(nextState)) - self.getQValue(state, action)
        
        # update weight: w_i = w_i + α * difference * f_i(s,a)
        for feature, value in features.items():
            self.weights[feature] = self.weights[feature] + self.alpha * difference * value

    def final(self, state):
        """Called at the end of each game."""
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # deb
            print("final weights after training:")
            print(self.weights)
