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

"""
part 2-2 & part 2-3
"""

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
        # Begin your code
        # Initialize Q-values as a util.Counter, which defaults all values to zero.
        # This is where all Q-values for state-action pairs will be stored.
        self.qValues = util.Counter()
        # End your code


    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        # Begin your code
        # return qValue base on state and action
        return self.qValues[(state, action)]
        # End your code


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        # Begin your code
        # Get all legal actions for the current state
        legalActions = self.getLegalActions(state)
        # If there are no legal actions, return 0.0
        if len(legalActions) == 0:
            return 0.0
        # Otherwise, return the maximum Q-value for the legal state
        return max([self.getQValue(state, action) for action in legalActions])
        # End your code

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        # Begin your code
        # Get all legal actions for the current state
        legalActions = self.getLegalActions(state)
        # If there are no legal actions, return None
        if not legalActions:
          return None
        # Otherwise, return the action with the maximum Q-value
        bestValue = self.computeValueFromQValues(state)
        bestActions = [action for action in legalActions if self.getQValue(state, action) == bestValue]
        return random.choice(bestActions)  
        # End your code

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
        # Begin your code
        # If there are no legal actions, return None
        if not legalActions:
          return None
        # With probability self.epsilon, take a random action
        if util.flipCoin(self.epsilon): 
          return random.choice(legalActions)
        # Otherwise, take the best policy action
        else: 
          return self.computeActionFromQValues(state)
        # End your code
        

    def update(self, state, action, nextState, reward):
        "*** YOUR CODE HERE ***"
        # Begin your code
        # Compute the sample, which is the reward plus the discounted value of the next state
        sample = reward + self.discount * self.computeValueFromQValues(nextState)
        # Update the Q-value for the current state and action
        self.qValues[(state, action)] = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * sample
        # End your code

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


"""
part 2-4
"""

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
        "*** YOUR CODE HERE ***"
        # Begin your code
        # get weights and feature
        # return the dot product of the weights and the features
        return self.getWeights() * self.featExtractor.getFeatures(state, action)
        # End your code

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        # Begin your code
        # Get the features for the current state and action
        features = self.featExtractor.getFeatures(state, action)
        # Compute the correction, which is the reward plus the discounted value of the next state 
        # minus the current Q-value for the state and action
        correction = reward + self.discount * self.computeValueFromQValues(nextState) - self.getQValue(state, action)
        # Update the weights for each feature
        for feature in features:
            self.weights[feature] += self.alpha * correction * features[feature]
        # End your code

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)
