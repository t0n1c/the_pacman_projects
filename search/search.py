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

from ..util import raiseNotDefined, ComparableMixin, Stack, Queue, PriorityQueue, SearchError
from ..game import Directions


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


class SearchNode(ComparableMixin):
    def __init__(self, state, action, path_cost, heuristic_cost, parent=None):
        self.state = state
        self.action = action
        self.path_cost = path_cost
        self.heuristic_cost = heuristic_cost
        self.cost = path_cost + heuristic_cost
        if parent is None:
            self.parent = self
        else:
            self.parent = parent

    def _actions_generator(self):
        while self.action != Directions.STOP:
            yield self
            self = self.parent
        return self

    def construct_actions_seq(self):
        return [parent.action for parent in self._actions_generator()][::-1]

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.state == other.state
        return False

    def __lt__(self, other):
        return self.cost < other.cost

    def __hash__(self):
        return hash(self.state)

    def __str__(self):
        pattern = '(State={0}, Action={1}, Path cost={2}, Heuristic cost={3}, Total cost={4})'
        return type(self).__name__ + pattern.format(self.state, self.action,
                                                    self.path_cost, self.heuristic_cost,
                                                    self.cost)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def is_valid_node(next_node, open_nodes, closed_nodes,
                  check_cost, only_closed):
    cost = next_node.cost
    if only_closed:
        return not next_node in closed_nodes
    # open is a superset of closed
    less_cost_open = lambda: cost < open_nodes[next_node]
    return not next_node in open_nodes or check_cost and less_cost_open()



def generic_search(problem, frontier_queue, heuristic=nullHeuristic,
                   check_cost=False, only_closed=False):

    frontier = frontier_queue()
    open_nodes = dict()
    closed_nodes = dict()
    start_state = problem.getStartState()
    start_node = SearchNode(start_state, Directions.STOP, 0, heuristic(start_state,problem))
    frontier.put(start_node)
    open_nodes[start_node] = start_node.cost

    while len(frontier) != 0:
        current_node = frontier.get()
        closed_nodes[current_node] = current_node.cost
        if problem.isGoalState(current_node.state):
            return current_node.construct_actions_seq()
        add_successors(problem, current_node, heuristic,
                       frontier, open_nodes, closed_nodes,
                       check_cost, only_closed)

    raise SearchError('No more states to search in. No solution found.')


def add_successors(problem, parent, heuristic, frontier,
                   open_nodes, closed_nodes, check_cost, only_closed):

    for state,action,step_cost in problem.getSuccessors(parent.state):
        path_cost = step_cost + parent.path_cost
        heuristic_cost = heuristic(state, problem)
        next_node = SearchNode(state, action, path_cost, heuristic_cost, parent)
        if is_valid_node(next_node, open_nodes, closed_nodes,
                         check_cost, only_closed):
            if not only_closed:
                open_nodes[next_node] = next_node.cost
            frontier.put(next_node)


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from ..game import Directions
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
    return generic_search(problem, Stack, only_closed=True)

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
