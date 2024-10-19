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
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        return childGameState.getScore()

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
    Your minimax agent (par1-1)
    """
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # Begin your code
        # Define the minimax function with parameters for the current agent, 
        # depth, game state, and a boolean for maximizing or not.
        def minimax(agentIndex, depth, gameState, maximizingPlayer): 
            # Base case: return the game state and its evaluation if the game is terminal state
            if gameState.isWin() or gameState.isLose() or (depth == 0 and agentIndex == 0):
                return (gameState,self.evaluationFunction(gameState))#return (state,score) pair
            # Calculate the next agent and depth based on 
            # the current agent and whether it's maximizing player's turn.
            nextAgent = (agentIndex + 1) % gameState.getNumAgents()
            nextDepth = depth - 1 if maximizingPlayer else depth
            # Get all possible actions for the current agent.
            actions = gameState.getLegalActions(agentIndex)
            
            if maximizingPlayer: #the turn of the maximizing player (Pacman)
                #next layer will be a min layer with the first ghost(index=1)
                arr = [minimax(1,nextDepth,gameState.getNextState(agentIndex,action),False) for action in actions]
                # compute the maximum score of successor states.
                return max(arr, key = lambda item: item[1])
            else: 
                if agentIndex < gameState.getNumAgents() - 1:# the turn of a min layer, current ghost is not the last ghost
                    #the next layser is still min layer
                    arr = [minimax(nextAgent, nextDepth,gameState.getNextState(agentIndex,action), False) for action in actions]
                    # compute the minimum score for non-final ghosts
                    return min(arr, key = lambda item : item[1])
                else:# the turn of a min layer, current ghost is the last ghost
                    #the next layer is max layer with pacman(index=0)
                    arr = [minimax(0, nextDepth,gameState.getNextState(agentIndex,action), True) for action in actions]
                    # compute the minimum score for final ghosts
                    return min(arr, key = lambda item : item[1])
                
        actions = gameState.getLegalActions(0) # Get legal actions for Pacman(INDEX = 0)
        #compute minimax for each action
        tup = [(action,minimax(1,self.depth - 1,gameState.getNextState(0,action),False)) for action in actions]
        bestAction, _ = max(tup, key = lambda item: item[1][1]) # choose the action with the highest score.
        return bestAction #return the action
        # End your code
    
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (part1-2)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # Begin your code
        # Define the expectimax function with parameters for the current agent, 
        # depth, game state, and a boolean for maximizing or not.
        def expectimax(agentIndex, depth, gameState, maximizingPlayer):
            # Base case: If the game is terminal state, return the evaluation.
            if gameState.isWin() or gameState.isLose() or (depth == 0 and agentIndex == 0):
                return self.evaluationFunction(gameState) #return score
            # Calculate the next agent and depth based on 
            # the current agent and whether it's maximizing player's turn.
            nextAgent = (agentIndex + 1) % gameState.getNumAgents()
            nextDepth = depth - 1 if maximizingPlayer else depth
            # Get all possible actions for the current agent.
            actions = gameState.getLegalActions(agentIndex)
            
            if maximizingPlayer: #the turn of the maximizing player (Pacman)
                #next layer will be a min layer with the first ghost(index=1)
                arr = [expectimax(1,nextDepth,gameState.getNextState(agentIndex,action),False) for action in actions]
                # compute the maximum score of successor states.
                return max(arr)
            else: 
                if agentIndex < gameState.getNumAgents() - 1:# the turn of a min layer, current ghost is not the last ghost
                    #the next layser is still min layer
                    arr = [expectimax(nextAgent, nextDepth,gameState.getNextState(agentIndex,action), False) for action in actions]
                    #take the average value over each action
                    return sum(arr) / len(actions)
                else: # the turn of a min layer, current ghost is the last ghost
                    #the next layer is max layer with pacman(index=0)
                    arr = [expectimax(0, nextDepth,gameState.getNextState(agentIndex,action), True) for action in actions]
                    #take the average value over each action
                    return sum(arr) / len(actions)
                
        actions = gameState.getLegalActions(0)# Get legal actions for Pacman(INDEX = 0)
        #compute expectimax for each action
        arr = [(action,expectimax(1,self.depth - 1,gameState.getNextState(0,action),False)) for action in actions]
        bestAction, _ = max(arr, key = lambda item: item[1]) # choose the action with the highest score.
        return bestAction #return the action
        # End your code 

better = scoreEvaluationFunction
