### MDP Value Iteration and Policy Iteration

import numpy as np
from riverswim import RiverSwim

np.set_printoptions(precision=3)

def bellman_backup(state, action, R, T, gamma, V):
    """
    Perform a single Bellman backup.

    Parameters
    ----------
    state: int
    action: int
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    gamma: float
    V: np.array (num_states)

    Returns
    -------
    backup_val: float
    """
    backup_val = None
    ############################
    ### START CODE HERE ###
    backup_val = R[state, action] + gamma*np.dot(T[state, action], V)
    ### END CODE HERE ###
    ############################

    return backup_val

def policy_evaluation(policy, R, T, gamma, tol=1e-3):
    """
    Compute the value function induced by a given policy for the input MDP
    Parameters
    ----------
    policy: np.array (num_states)
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    gamma: float
    tol: float

    Returns
    -------
    value_function: np.array (num_states)
    """
    num_states, _ = R.shape
    value_function = None

    ############################
    ### START CODE HERE ###
    value_function = np.zeros(num_states)  # init V0=0

    while True:
        delta = 0  # track the max change in value func
        new_v = np.copy(value_function)  # Copy new value func
        for state in range(num_states):
            # Get action
            action = int(policy[state])
            # Compute BV for current s, a
            backup_val = bellman_backup(state, action, R, T, gamma, value_function)
            # Calculate change in value for s
            delta = max(delta, abs(backup_val-value_function[state]))
            new_v[state] = backup_val  
        value_function = new_v  # Update final value iteration
        
        # Convergence
        if delta < tol:
            break
    ### END CODE HERE ###
    ############################
    return value_function


def policy_improvement(R, T, V_policy, gamma):
    """
    Given the value function induced by a given policy, perform policy improvement
    Parameters
    ----------
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    V_policy: np.array (num_states)
    gamma: float

    Returns
    -------
    new_policy: np.array (num_states)
    """
    num_states, num_actions = R.shape
    new_policy = None

    ############################
    ### START CODE HERE ###
    new_policy = np.zeros(num_states, dtype=int)
    for state in range(num_states):
        # Comput BV for all actions in s
        action_values = np.zeros(num_actions)
        for action in range(num_actions):
            action_values[action] = bellman_backup(state, action, R, T, gamma, V_policy)
        
        # Choose action that maximizes value
        new_policy[state] = np.argmax(action_values)
    ### END CODE HERE ###
    ############################
    return new_policy


def policy_iteration(R, T, gamma, tol=1e-3):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.
    Parameters
    ----------
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)

    Returns
    -------
    V_policy: np.array (num_states)
    policy: np.array (num_states)
    """
    num_states, num_actions = R.shape
    V_policy = None
    policy = None
    ############################
    ### START CODE HERE ###
    # Init random policy
    policy = np.random.choice(num_actions, size=num_states)

    while True:  # Same as L1 Norm???
        # Policy evaluation for current
        V_policy = policy_evaluation(policy, R, T, gamma, tol)
        # Compute new using value function
        new_policy = policy_improvement(R, T, V_policy, gamma)
        # Check convergence
        if np.array_equal(policy, new_policy):
            break
        policy = new_policy  # Update
    ### END CODE HERE ###
    ############################
    return V_policy, policy


def value_iteration(R, T, gamma, tol=1e-3):
    """Runs value iteration.
    Parameters
    ----------
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)

    Returns
    -------
    value_function: np.array (num_states)
    policy: np.array (num_states)
    """
    num_states, num_actions = R.shape
    value_function = None
    policy = None
    ############################
    ### START CODE HERE ###
    value_function = np.zeros(num_states)  # Init V0=0

    while True:
        delta = 0
        new_v = np.zeros(num_states)

        # Update value using Bellman optimality
        for state in range(num_states):
            action_values = np.zeros(num_actions)
            for action in range(num_actions):
                action_values[action] = bellman_backup(state, action, R, T, gamma, value_function)
            # Take max across all actions in s
            new_v[state] = np.max(action_values)
            # Check convergence
            delta = max(delta, abs(new_v[state]-value_function[state]))
        value_function = new_v  # Update
        if delta < tol:
            break
    
    # Derive optimal policy from optimal value function    
    policy = policy_improvement(R, T, value_function, gamma)
    ### END CODE HERE ###
    ############################
    return value_function, policy


# Edit below to run policy and value iteration on different configurations
# You may change the parameters in the functions below
if __name__ == "__main__":
    SEED = 1234

    RIVER_CURRENT = 'STRONG'
    assert RIVER_CURRENT in ['WEAK', 'MEDIUM', 'STRONG']
    env = RiverSwim(RIVER_CURRENT, SEED)

    R, T = env.get_model()
    discount_factor = 0.25

    print("\n" + "-" * 25 + "\nBeginning Policy Iteration\n" + "-" * 25)

    V_pi, policy_pi = policy_iteration(R, T, gamma=discount_factor, tol=1e-3)
    print(V_pi)
    print([['L', 'R'][a] for a in policy_pi])

    print("\n" + "-" * 25 + "\nBeginning Value Iteration\n" + "-" * 25)

    V_vi, policy_vi = value_iteration(R, T, gamma=discount_factor, tol=1e-3)
    print(V_vi)
    print([['L', 'R'][a] for a in policy_vi])