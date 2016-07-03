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



import random
import os
import sys
from collections import deque

from ..game import Directions, Agent
from .. import util
from ..util import euclidean_distance as get_distance, SearchError, all_argmax

from ..search.search import uniformCostSearch, aStarSearch
from ..search.searchAgents import AnyFoodSearchProblem, PositionSearchProblem, manhattanHeuristic


# THR stands for THRESHOLD
DANGER_THR = 1.0
CAPSULE_THR = 2.0
MAX_SCORE = 99999999
MIN_SCORE = -MAX_SCORE


# https://inst.eecs.berkeley.edu/~cs188/fa10/slides/FA10%20cs188%20lecture%202%20--%20uninformed%20search%20(6PP).pdf
#http://www.cs.cmu.edu/~sandholm/cs15-381/Agents.ppt




class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def __init__(self):
        super().__init__()
        self.pre_actions = deque()

    def getAction(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a game_state and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states

        legal_moves = game_state.getLegalActions()
        # Choose one of the best actions
        scores = [self.evaluationFunction(game_state, action) for action in legal_moves]

        if len(self.pre_actions) > 0: # fixed actions mode
            if self.pre_actions[0] in dangerous_actions(legal_moves, scores):
                self.pre_actions = deque()
            else:
                return self.pre_actions.popleft()

        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = random.choice(best_indices) # Pick randomly among the best
        return legal_moves[chosen_index]

    def evaluationFunction(self, game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)

        successor = game_state.generatePacmanSuccessor(action) # game state successor
        pacman_pos = successor.getPacmanPosition()
        ghosts_info = [(g.getPosition(), g.scaredTimer) for g in successor.getGhostStates()]

        current_capsules = game_state.getCapsules()
        kwargs_psp = dict(gameState=game_state, warn=False)
        scared_ghosts_ = scared_ghosts(ghosts_info)

        if not is_pacman_safe(pacman_pos, ghosts_info):
            return float('-Inf')

        elif action == 'Stop':
            return MIN_SCORE

        elif is_any_capsule_close(pacman_pos, current_capsules):
            self.pre_actions = deque(capsule_actions(pacman_pos, current_capsules, **kwargs_psp))
            return MAX_SCORE

        elif len(scared_ghosts_) > 0:
            self.pre_actions = deque(scared_ghost_actions(scared_ghosts_, **kwargs_psp))
            return MAX_SCORE

        else:
            return -distance_to_closest_food(pacman_pos, successor, game_state.getFood())



def is_pacman_safe(pacman_pos, ghosts_info):
    check_timers = (scared_timer > 0 for _,scared_timer in ghosts_info)
    check_pos = (get_distance(pacman_pos, p2) > DANGER_THR for p2,_ in ghosts_info)
    return all(cond1 or cond2 for cond1,cond2 in zip(check_timers,check_pos))


def get_distances_to_ghosts(pacman_pos, ghosts_info):
    distances = list()
    for pos,_ in ghosts_info:
        distance = get_distance(pacman_pos, pos)
        if distance < DANGER_THR+1:
            distances.append(distance)
    return distances


def is_any_capsule_close(pacman_pos, capsules, threshold=CAPSULE_THR):
    return any([get_distance(pacman_pos, c_pos) <= threshold for c_pos in capsules])


def capsule_actions(pacman_pos, capsules, **kwargs_psp):
    """Return a set of actions that leads to the closest capsule.
    """
    capsule_pos = closest_capsule(pacman_pos, capsules)
    problem = PositionSearchProblem(goal=capsule_pos, **kwargs_psp)
    return aStarSearch(problem, manhattanHeuristic)


def scared_ghosts(ghosts_info):
    return [tuple(map(int, xy)) for xy,timer in ghosts_info if timer > 0.1]

def scared_ghost_actions(scared_ghosts, **kwargs_psp):
    """Closest scared ghost set of actions"""
    problems = [PositionSearchProblem(goal=xy, **kwargs_psp) for xy in scared_ghosts]
    ghostbuster_ways = [aStarSearch(problem, manhattanHeuristic) for problem in problems]
    return min(ghostbuster_ways, key=len)


def get_first_food_actions(game_state):
    problem = AnyFoodSearchProblem(game_state)
    try:
        return uniformCostSearch(problem)
    except SearchError:
        return list()


def closest_capsule(pacman_pos, capsules):
    all_distances = [(get_distance(pacman_pos, c_pos)) for c_pos in capsules]
    return capsules[all_distances.index(min(all_distances))]


def dangerous_actions(legal_moves, scores):
    return [m for m,s in zip(legal_moves,scores) if s == float('-Inf')]


def distance_to_closest_food(pacman_pos, successor, current_maze_food):
    new_x, new_y = pacman_pos
    if current_maze_food[new_x][new_y]:
        return 0
    return len(get_first_food_actions(successor))



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

    def getAction(self, game_state):
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
        """
        scores = [self.min_value(s, 1, self.depth) for s in get_successors(game_state, 0)]
        actions = game_state.getLegalActions(0)
        return random.choice([actions[i] for i in all_argmax(scores)])

    def min_value(self, game_state, agent_index, depth):
        if depth == 0 or len(game_state.getLegalActions(agent_index)) == 0:
            return self.evaluationFunction(game_state)
        successors = get_successors(game_state, agent_index)
        minmax_fn, *args = self._set_up_minmax(game_state, agent_index, depth)
        return min([minmax_fn(s, *args) for s in successors] + [float('Inf')])

    def max_value(self, game_state, agent_index, depth):
        if depth == 0 or len(game_state.getLegalActions(agent_index)) == 0:
            return self.evaluationFunction(game_state)
        successors = get_successors(game_state, agent_index)
        return max([self.min_value(s, 1, depth) for s in successors] + [float('-Inf')])

    def _set_up_minmax(self, game_state, agent_index, depth):
        if agent_index == game_state.getNumAgents() - 1:
            minmax_fn = self.max_value
            next_agent_index = 0
            depth -= 1
        else:
            minmax_fn = self.min_value
            next_agent_index = agent_index + 1
        return minmax_fn, next_agent_index, depth


def get_successors(current_state, agent_index):
    for action in current_state.getLegalActions(agent_index):
        yield current_state.generateSuccessor(agent_index, action)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, game_state):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        successors = get_successors(game_state, 0)
        alpha, beta = float('-Inf'), float('Inf')
        current_max = float('-Inf')
        scores = list()
        for succ in successors:
            min_v = self.min_value(succ, 1, self.depth, alpha, beta)
            scores.append(min_v)
            current_max = max(current_max, min_v)
            alpha = max(alpha, current_max)
        actions = game_state.getLegalActions(0)
        return random.choice([actions[i] for i in all_argmax(scores)])


    def min_value(self, game_state, agent_index, depth, alpha, beta):
        if depth == 0 or len(game_state.getLegalActions(agent_index)) == 0:
            return self.evaluationFunction(game_state)
        successors = get_successors(game_state, agent_index)
        minmax_fn, next_agent_index, depth = self._set_up_minmax(game_state, agent_index, depth)
        current_min = float('Inf')
        for succ in successors:
            current_min = min(current_min, minmax_fn(succ, next_agent_index, depth, alpha, beta))
            if current_min < alpha:
                return current_min
            beta = min(beta, current_min)
        return current_min


    def max_value(self, game_state, agent_index, depth, alpha, beta):
        if depth == 0 or len(game_state.getLegalActions(agent_index)) == 0:
            return self.evaluationFunction(game_state)
        current_max = float('-Inf')
        for succ in get_successors(game_state, agent_index):
            current_max = max(current_max, self.min_value(succ, 1, depth, alpha, beta))
            if current_max > beta:
                return current_max
            alpha = max(alpha, current_max)
        return current_max


    def _set_up_minmax(self, game_state, agent_index, depth):
        if agent_index == game_state.getNumAgents() - 1:
            minmax_fn = self.max_value
            next_agent_index = 0
            depth -= 1
        else:
            minmax_fn = self.min_value
            next_agent_index = agent_index + 1
        return minmax_fn, next_agent_index, depth

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, game_state):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        root_successors = zip(game_state.getLegalActions(0), get_successors(game_state, 0))
        weight = 1.0 / len(game_state.getLegalActions(0))
        scores = [(a, weight * self.expected_value(s, 1, self.depth)) for a,s in root_successors]
        highest_score = max(scores, key=lambda pair: pair[1])[1]
        return random.choice([(a,score) for a,score in scores if score == highest_score])[0]


    def expected_value(self, game_state, agent_index, depth):
        legal_actions = game_state.getLegalActions(agent_index)
        if depth == 0 or len(legal_actions) == 0:
            return self.evaluationFunction(game_state)
        successors = get_successors(game_state, agent_index)
        minmax_fn, next_agent_index, depth = self._set_up_minmax(game_state, agent_index, depth)
        return sum([1.0/len(legal_actions) * minmax_fn(s, next_agent_index, depth) for s in successors])


    def max_value(self, game_state, agent_index, depth):
        if depth == 0 or len(game_state.getLegalActions(agent_index)) == 0:
            return self.evaluationFunction(game_state)
        successors = get_successors(game_state, agent_index)
        return max([self.expected_value(s, 1, depth) for s in successors] + [float('-Inf')])


    def _set_up_minmax(self, game_state, agent_index, depth):
        if agent_index == game_state.getNumAgents() - 1:
            minmax_fn = self.max_value
            next_agent_index = 0
            depth -= 1
        else:
            minmax_fn = self.expected_value
            next_agent_index = agent_index + 1
        return minmax_fn, next_agent_index, depth


def betterEvaluationFunction(game_state):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

    """
    methods_names = ['getPacmanPosition','getFood','getGhostStates']
    new_info_state = [getattr(game_state, name)() for name in methods_names]
    pacman_pos, food, ghost_states = new_info_state

    ghosts_info = [(g.getPosition(),g.scaredTimer) for g in ghost_states]
    current_capsules = game_state.getCapsules()
    kwargs_psp = dict(gameState=game_state, warn=False)
    scared_ghosts = [tuple(map(int, xy)) for xy,timer in ghosts_info if timer > 0.1]

    if not is_pacman_safe(pacman_pos, ghosts_info):
        distances = get_distances_to_ghosts(pacman_pos, ghosts_info)
        if 0.0 in distances:
            return MIN_SCORE
        return MIN_SCORE + sum(distances)/len(distances)

    elif len(scared_ghosts) > 0:
        return 100 + game_state.getScore() - len(scared_ghost_actions(scared_ghosts, **kwargs_psp))

    else:
        return game_state.getScore() - (distance_to_closest_food(pacman_pos, game_state,game_state.getFood()))**0.5

# Abbreviation
better = betterEvaluationFunction

