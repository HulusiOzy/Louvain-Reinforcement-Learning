#!/usr/bin/env python3

import copy
import random
import numpy as np
import networkx as nx


CELL_TYPES = {".": "floor", "#": "wall", "S": "start", "G": "goal"}
ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # UP, DOWN, LEFT, RIGHT
ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}


class GridWorld:
    """
    Discrete gridworld loaded from a text file.
    Compatible with agent.py: step(state, action) -> (next_state, reward, terminal).
    """

    def __init__(self, grid_file, movement_penalty=-0.001, goal_reward=1.0):
        self.grid = np.loadtxt(grid_file, comments="//", dtype=str)
        self.movement_penalty = movement_penalty
        self.goal_reward = goal_reward

        self.initial_states = []
        self.terminal_states = []
        self.state_space = set()
        self.options = []  # populated externally before agent.run()

        # Parse the grid.
        for y in range(self.grid.shape[0]):
            for x in range(self.grid.shape[1]):
                cell = self.grid[y, x]
                if cell not in CELL_TYPES:
                    continue
                if CELL_TYPES[cell] == "wall":
                    continue
                self.state_space.add((y, x))
                if CELL_TYPES[cell] == "start":
                    self.initial_states.append((y, x))
                elif CELL_TYPES[cell] == "goal":
                    self.terminal_states.append((y, x))

        # Pre-compute transition table for speed.
        self.transitions = {}
        for s in self.state_space:
            if self.is_terminal(s):
                continue
            for a in range(4):
                dy, dx = ACTIONS[a]
                ny, nx_ = s[0] + dy, s[1] + dx
                # Bounce off walls.
                if self.grid[ny, nx_] in CELL_TYPES and CELL_TYPES[self.grid[ny, nx_]] == "wall":
                    ns = s
                else:
                    ns = (ny, nx_)
                # Reward.
                if ns in self.terminal_states:
                    r = self.goal_reward + self.movement_penalty
                else:
                    r = self.movement_penalty
                self.transitions[(s, a)] = (ns, r, ns in self.terminal_states)

        self.current_state = None

    def reset(self, state=None):
        if state is None:
            self.current_state = random.choice(self.initial_states)
        else:
            self.current_state = state
        return self.current_state

    def step(self, state, action):
        """Functional step: takes explicit state, returns (next_state, reward, terminal)."""
        ns, r, done = self.transitions[(state, action)]
        self.current_state = ns
        return ns, r, done

    def is_terminal(self, state):
        return state in self.terminal_states

    def get_actions(self):
        return [0, 1, 2, 3]

    def generate_stg(self, directed=True):
        """Build the state-transition graph (networkx DiGraph).
        Nodes are (y, x) tuples, edges are possible transitions."""
        if directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()

        for s in self.state_space:
            G.add_node(s)
            if self.is_terminal(s):
                continue
            for a in range(4):
                ns, _, _ = self.transitions[(s, a)]
                if ns != s:  # skip self-loops from wall bounces
                    G.add_edge(s, ns)

        return G
