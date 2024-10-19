# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections


"""
part 2-1
"""

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # Begin your code
        # Iterate over each iteration.
        for i in range(self.iterations): 
            # Keep a copy of the current values to use for updates.
            previous_value = self.values.copy() 
            # Get all the states in the MDP.
            states = self.mdp.getStates()
            # Iterate over each state to update its value.
            for state in states:
                # If the state is terminal, its value is zero.
                if self.mdp.isTerminal(state): 
                    self.values[state] = 0
                    continue
                # Initialize max_value to a very small number.
                max_value = float('-inf')
                # Get all possible actions from the current state.
                actions = self.mdp.getPossibleActions(state)
                # Iterate over each action to calculate its value.
                for action in actions: 
                    total = 0
                     # Get the transition states and probabilities of the current action.
                    transitions = self.mdp.getTransitionStatesAndProbs(state, action)
                    # Sum up the value of all transitions from the current state-action pair.
                    for (nextState, prob) in transitions : 
                        total += prob * (self.mdp.getReward(state, action, nextState) + self.discount*previous_value[nextState]) 
                    # Update max_value if the calculated total is greater.
                    if total > max_value:
                        max_value = total
                # Update the value of the state with the maximum value found.
                self.values[state] = max_value
        # End your code


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # Begin your code
        q_value = 0 # Initialize Q-value
        # Retrieve all transitions and their probabilities for the given state-action pair
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
         # Sum up the value for each transition
        for (nextState, prob) in transitions: 
             # Each part contributes to the Q-value: probability * (immediate reward + discounted future value)
            q_value += prob * (self.mdp.getReward(state, action, nextState)  + self.discount*self.values[nextState])
        return q_value  # Return the calculated Q-value
        # End your code

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # Begin your code
        #check for terminal
        # If the state is terminal, there are no actions to take, return None.
        if self.mdp.isTerminal(state): 
            return None
        q_value = util.Counter() # Initialize a Counter to store Q-values for each action.
        actions = self.mdp.getPossibleActions(state)  # Get all legal actions for the state.
        # Compute the Q-value for each action and store in the Counter.
        for action in actions:
            q_value[action] = self.computeQValueFromValues(state, action)
        # Return the action with the highest Q-value, breaking ties arbitrarily.
        return q_value.argMax()
        # End your code

    def getPolicy(self, state):
        """
        The policy is the best action in the given state
        according to the values computed by value iteration.
        You may break ties any way you see fit.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return None.
        """
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        """
        The q-value of the state action pair
        (after the indicated number of value iteration
        passes).  Note that value iteration does not
        necessarily create this quantity and you may have
        to derive it on the fly.
        """
        return self.computeQValueFromValues(state, action)
