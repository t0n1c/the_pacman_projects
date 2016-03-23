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

from .util import raiseNotDefined, ComparableMixin, Stack, Queue, PriorityQueue
from .game import Directions


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
        raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        raiseNotDefined()


class SearchFailure(Exception):
    pass

class SearchNode(ComparableMixin):
    def __init__(self, state, action, node_cost, heuristic_cost, parent=None):
        self.state = state
        self.action = action
        self.node_cost = node_cost
        self.heuristic_cost = heuristic_cost
        self.cost = node_cost + heuristic_cost
        if parent is None:
            self.parent = self
        else:
            self.parent = parent

    def actions_generator(self):
        while self.action != Directions.STOP:
            yield self
            self = self.parent
        return self

    def construct_actions_path(self):
        return [parent.action for parent in self.actions_generator()][::-1]

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.state == other.state
        return False

    def __lt__(self, other):
        return self.cost < other.cost

    def __hash__(self):
        return hash(self.state)

    def __str__(self):
        return str(self.state)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def check_search_node(next_node, open_nodes_cost, closed_nodes_cost,
                      check_cost, check_only_closed):
    cost = next_node.cost
    if check_only_closed:
        return not next_node in closed_nodes_cost
    # open is a superset of closed
    less_cost_open = lambda: cost < open_nodes_cost[next_node]
    return not next_node in open_nodes_cost or check_cost and less_cost_open()



def generic_search(problem, data_structure, heuristic=nullHeuristic, check_cost=False,
                   check_only_closed=False):

    frontier = data_structure()
    open_nodes_cost = dict()
    closed_nodes_cost = dict()
    start_state = problem.getStartState()
    start_node = SearchNode(start_state,Directions.STOP,0,heuristic(start_state,problem))
    frontier.put(start_node)
    open_nodes_cost[start_node] = start_node.cost

    while len(frontier) != 0:
        current_node = frontier.get()
        closed_nodes_cost[current_node] = current_node.cost
        if problem.isGoalState(current_node.state):
            break
        successors = problem.getSuccessors(current_node.state)
        for state,action,cost in successors:
            total_cost = cost + current_node.node_cost
            heuristic_cost = heuristic(state, problem)
            next_node = SearchNode(state, action, total_cost, heuristic_cost, current_node)
            args = next_node, open_nodes_cost, closed_nodes_cost, check_cost, check_only_closed
            if check_search_node(*args):
                open_nodes_cost[next_node] = next_node.cost
                frontier.put(next_node)
    else:
        raise SearchFailure('No more states to search in. No solution found.')

    return current_node.construct_actions_path()

def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from .game import Directions
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

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    return generic_search(problem, Stack, check_only_closed=True)

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    return generic_search(problem, Queue)

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    return generic_search(problem, PriorityQueue, check_cost=True)

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    return generic_search(problem, PriorityQueue, heuristic, check_cost=True)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
