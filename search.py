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
from game import Actions


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
        """Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999. This is implemented for you."""
        if actions is None: return 999999
        x, y = self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999  # Check for walls
        return len(actions)  # Cost is the number of actions


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem: SearchProblem):
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
    # Initialize a stack for DFS
    stack = util.Stack()

    # Start with the initial state and an empty path
    stack.push((problem.getStartState(), []))

    # Set to keep track of visited nodes
    visited = set()

    while not stack.isEmpty():
        # Pop a state and path from the stack
        currentState, path = stack.pop()

        # Check if this state is the goal
        if problem.isGoalState(currentState):
            return path  # Return the path found

        # Skip if state is visited
        if currentState in visited:
            continue

        # Mark the state as visited
        visited.add(currentState)

        # For each successor of the current state
        for successor, action, stepCost in problem.getSuccessors(currentState):
            if successor not in visited:
                # Push the successor state and the path to reach it onto the stack
                stack.push((successor, path + [action]))

    # Return empty list if no path is found
    return []


def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    queue = util.Queue()
    visited = set()
    queue.push((problem.getStartState(), []))

    while not queue.isEmpty():
        state, actions = queue.pop()

        if problem.isGoalState(state):
            return actions

        if state not in visited:
            visited.add(state)
            for successor, action, stepCost in problem.getSuccessors(state):
                if successor not in visited:
                    queue.push((successor, actions + [action]))

    return []


def uniformCostSearch(problem):
    # Priority queue with state and cost
    priorityQueue = util.PriorityQueue()
    priorityQueue.push((problem.getStartState(), [], 0), 0)
    visited = set()

    while not priorityQueue.isEmpty():
        state, actions, currentCost = priorityQueue.pop()

        if problem.isGoalState(state):
            return actions

        if state not in visited:
            visited.add(state)
            for successor, action, stepCost in problem.getSuccessors(state):
                if successor not in visited:
                    newCost = currentCost + stepCost
                    priorityQueue.push((successor, actions + [action], newCost), newCost)

    return []


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    priorityQueue = util.PriorityQueue()
    startState = problem.getStartState()
    priorityQueue.push((startState, [], 0), heuristic(startState, problem))
    visited = set()

    while not priorityQueue.isEmpty():
        state, actions, currentCost = priorityQueue.pop()

        if problem.isGoalState(state):
            return actions

        if state not in visited:
            visited.add(state)
            for successor, action, stepCost in problem.getSuccessors(state):
                if successor not in visited:
                    newCost = currentCost + stepCost
                    heuristicCost = heuristic(successor, problem)
                    totalCost = newCost + heuristicCost
                    priorityQueue.push((successor, actions + [action], newCost), totalCost)

    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
