#!/usr/bin/env python3

import math
import random
from collections import defaultdict


# Utilities

def discounted_return(rewards, gamma):
    total = 0.0
    gamma_power = 1.0
    for r in rewards:
        total += r * gamma_power
        gamma_power *= gamma
    return total


# Option: the data structure. Holds initiation set, termination condition,
# and a trained policy (populated by OptionTrainer, frozen during agent run).

class Option:
    def __init__(self, option_id, initiation_set, termination_fn, level=0):
        self.id = option_id
        self.I = initiation_set       # set of state IDs, or None = everywhere
        self.beta = termination_fn     # state -> float in [0,1]
        self.level = level
        self.trained_policy = {}       # state -> sub-option/primitive, filled by trainer
        self.sub_options = []          # set by train_hierarchy, used as fallback

    def initiation(self, s):
        return self.I is None or s in self.I

    def termination(self, s):
        return self.beta(s)

    def policy(self, s, test=False):
        """Returns the sub-option (or primitive) this option selects in state s.
        Falls back to random available sub-option for untrained states."""
        result = self.trained_policy.get(s)
        if result is not None:
            return result
        # Fallback: pick random sub-option that can initiate here.
        available = [o for o in self.sub_options if o.initiation(s)]
        if available:
            return random.choice(available)
        return None

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Option) and self.id == other.id

    def __repr__(self):
        return f"Option({self.id})"


class PrimitiveOption(Option):
    """A single primitive action wrapped as an option.
    Terminates immediately. Policy always returns its own action."""

    def __init__(self, action_id):
        super().__init__(
            option_id=f"prim_{action_id}",
            initiation_set=None,
            termination_fn=lambda s: 1.0,
            level=-1,
        )
        self.action = action_id

    def policy(self, s, test=False):
        return self.action

    def __repr__(self):
        return f"PrimitiveOption({self.action})"


# OptionTrainer: offline training of option policies.
#
# For each Louvain option (source_cluster -> target_cluster), we spin up
# a small training loop where the agent starts in the source cluster and
# learns to reach the target cluster. The reward structure is:
#   +1.0  for reaching the target cluster
#   -1.0  for leaving the source cluster (if not allowed) or hitting terminal
#   -0.001 per step otherwise
#
# This matches option_trainers.py lines 90-107 in the authors' code.
# The result is a Q-table that we convert into a greedy policy dict
# stored on the Option object.

class OptionTrainer:
    def __init__(self, env, epsilon=0.2, alpha=0.4, gamma=1.0,
                 max_steps=100_000, max_episode_steps=250):
        self.env = env
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.max_steps = max_steps
        self.max_episode_steps = max_episode_steps

    def train(self, option, sub_options, initiation_states,
              target_states, can_leave_initiation=False):
        """Train an option's policy to navigate from source to target cluster.

        Args:
            option: the Option whose trained_policy we're populating.
            sub_options: list of options/primitives this option can call.
            initiation_states: set of states in the source cluster.
            target_states: set of states in the target cluster.
            can_leave_initiation: whether agent may leave source cluster en route.

        After training, option.trained_policy maps states to sub-options.
        """
        q = defaultdict(float)

        def available(s):
            return [o for o in sub_options if o.initiation(s)]

        def select(s):
            opts = available(s)
            if not opts:
                return None
            if random.random() < self.epsilon:
                return random.choice(opts)
            max_q = max(q[(hash(s), hash(o))] for o in opts)
            best = [o for o in opts if q[(hash(s), hash(o))] == max_q]
            return random.choice(best)

        # Filter out terminal states as starting points.
        start_states = [s for s in initiation_states
                        if not self.env.is_terminal(s)]
        if not start_states:
            return

        t = 0
        while t < self.max_steps:
            s = self.env.reset(random.choice(start_states))
            ep_steps = 0
            terminal = False

            while not terminal:
                o = select(s)
                if o is None:
                    break

                # Execute sub-option, collecting trajectory.
                states = [s]
                rewards = []
                option_done = False

                while not option_done and not terminal:
                    # Walk down to a primitive action.
                    prim = self._get_primitive(s, o)
                    if prim is None:
                        break
                    s_next, r_env, done = self.env.step(s, prim)
                    t += 1
                    ep_steps += 1

                    # Shaped reward for option training.
                    if s_next in target_states:
                        r = 1.0
                        terminal = True
                    elif done:
                        r = -1.0
                        terminal = True
                    elif (not can_leave_initiation
                          and s_next not in initiation_states
                          and s_next not in target_states):
                        r = -1.0
                        terminal = True
                    else:
                        r = -0.001

                    option_done = random.random() < o.termination(s_next)

                    states.append(s_next)
                    rewards.append(r)
                    s = s_next

                    if ep_steps > self.max_episode_steps or t >= self.max_steps:
                        break

                # Macro-Q update over the trajectory (1-step by default).
                if rewards:
                    term_state = states[-1]
                    for i in range(len(states) - 1):
                        old = q[(hash(states[i]), hash(o))]
                        dr = discounted_return(rewards[i:], self.gamma)
                        k = len(rewards) - i
                        if not terminal:
                            next_q = max(
                                (q[(hash(term_state), hash(op))]
                                 for op in available(term_state)),
                                default=0,
                            )
                        else:
                            next_q = 0
                        q[(hash(states[i]), hash(o))] = old + self.alpha * (
                            dr + math.pow(self.gamma, k) * next_q - old
                        )
                        break  # 1-step update

        # Extract greedy policy from trained Q-table.
        for s in initiation_states:
            if self.env.is_terminal(s):
                continue
            opts = available(s)
            if not opts:
                continue
            max_q = max(q[(hash(s), hash(o))] for o in opts)
            best = [o for o in opts if q[(hash(s), hash(o))] == max_q]
            option.trained_policy[s] = random.choice(best)

    def _get_primitive(self, s, option):
        """Walk down the hierarchy until we hit a primitive action."""
        if isinstance(option, PrimitiveOption):
            return option.action
        sub = option.policy(s)
        if sub is None:
            return None
        return self._get_primitive(s, sub)


# OptionAgent: online task learning using pre-trained options.
#
# Stack-based execution model (no recursion).
# At each primitive step:
#   - append state/reward to every active option's trajectory
#   - check termination top-down
#   - on termination: macro-Q + intra-option update, then pop
#
# Matches simpleoptions/options_agent.py

class OptionAgent:
    def __init__(self, env, options, epsilon=0.1, macro_alpha=0.4,
                 intra_alpha=0.4, gamma=1.0, default_q=0.0, n_step=False):
        self.env = env
        self.options = options  # all options including primitives
        self.epsilon = epsilon
        self.macro_alpha = macro_alpha
        self.intra_alpha = intra_alpha
        self.gamma = gamma
        self.n_step = n_step
        self.q_table = defaultdict(lambda: default_q)

        # Execution stack: parallel lists tracking active option hierarchy.
        self.executing_options = []
        self.executing_options_states = []
        self.executing_options_rewards = []

    def available_options(self, s):
        return [o for o in self.options if o.initiation(s)]

    def select_action(self, s, test=False):
        """Stack empty: epsilon-greedy over top-level Q-table.
        Stack non-empty: follow topmost option's trained policy."""
        if not self.executing_options:
            opts = self.available_options(s)
            if not test and random.random() < self.epsilon:
                return random.choice(opts)
            q_vals = [self.q_table[(hash(s), hash(o))] for o in opts]
            max_q = max(q_vals)
            best = [opts[i] for i, q in enumerate(q_vals) if q == max_q]
            return random.choice(best)
        else:
            return self.executing_options[-1].policy(s, test)

    def macro_q_learn(self, states, rewards, option):
        """Trajectory-based macro-Q update.
        With n_step=False, only updates Q(states[0], option)."""
        term_state = states[-1]

        for i in range(len(states) - 1):
            old = self.q_table[(hash(states[i]), hash(option))]
            dr = discounted_return(rewards[i:], self.gamma)
            k = len(rewards) - i

            if not self.env.is_terminal(term_state):
                next_q = max(
                    self.q_table[(hash(term_state), hash(o))]
                    for o in self.available_options(term_state)
                )
            else:
                next_q = 0

            self.q_table[(hash(states[i]), hash(option))] = old + self.macro_alpha * (
                dr + math.pow(self.gamma, k) * next_q - old
            )

            if not self.n_step:
                break

    def intra_option_learn(self, states, rewards, executed_option, higher_option=None):
        """Update Q-values for all options whose policy agrees with executed_option.
        Uses (1-beta)*Q_continue + beta*Q_terminate decomposition for the bootstrap."""
        term_state = states[-1]

        for i in range(len(states) - 1):
            init_state = states[i]

            for other in self.available_options(init_state):
                # Skip parent — it gets its own macro-Q update.
                if higher_option is not None and hash(other) == hash(higher_option):
                    continue
                # Only update options that would have picked the same sub-option.
                if hash(other.policy(init_state)) != hash(executed_option):
                    continue

                old = self.q_table[(hash(init_state), hash(other))]
                dr = discounted_return(rewards[i:], self.gamma)
                k = len(rewards) - i

                if not self.env.is_terminal(term_state):
                    beta = other.termination(term_state)
                    q_term = beta * max(
                        self.q_table[(hash(term_state), hash(o))]
                        for o in self.available_options(term_state)
                    )
                    q_cont = (1 - beta) * self.q_table[
                        (hash(term_state), hash(other))
                    ]
                else:
                    q_term = 0
                    q_cont = 0

                self.q_table[(hash(init_state), hash(other))] = old + self.intra_alpha * (
                    dr + math.pow(self.gamma, k) * (q_cont + q_term) - old
                )

            if not self.n_step:
                break

    def _pop_and_update(self):
        """Pop topmost option, perform both learning updates."""
        self.macro_q_learn(
            self.executing_options_states[-1],
            self.executing_options_rewards[-1],
            self.executing_options[-1],
        )
        self.intra_option_learn(
            self.executing_options_states[-1],
            self.executing_options_rewards[-1],
            self.executing_options[-1],
            self.executing_options[-2] if len(self.executing_options) > 1 else None,
        )
        self.executing_options.pop()
        self.executing_options_states.pop()
        self.executing_options_rewards.pop()

    def run(self, num_epochs, epoch_length):
        """Main training loop. Returns cumulative reward per epoch."""
        num_steps = num_epochs * epoch_length
        epoch_rewards = [0.0] * num_epochs
        t = 0

        while t < num_steps:
            s = self.env.reset()
            terminal = False

            while not terminal and t < num_steps:
                selected = self.select_action(s)

                # Any Option (including PrimitiveOption): push onto stack.
                # Next iteration, its policy will be queried.
                if isinstance(selected, Option):
                    self.executing_options.append(selected)
                    self.executing_options_states.append([s])
                    self.executing_options_rewards.append([])
                    continue

                # Raw action (returned by PrimitiveOption.policy): step env.
                s_next, r, terminal = self.env.step(s, selected)
                epoch_rewards[t // epoch_length] += r
                t += 1

                # Record transition for every active option on the stack.
                for i in range(len(self.executing_options)):
                    self.executing_options_states[i].append(s_next)
                    self.executing_options_rewards[i].append(r)

                s = s_next

                # Check termination top-down. Pop and update as needed.
                while (self.executing_options
                       and random.random() < self.executing_options[-1].termination(s)):
                    self._pop_and_update()

            # Episode ended. Flush the entire stack.
            while self.executing_options:
                self._pop_and_update()

        return epoch_rewards
