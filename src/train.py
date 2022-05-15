import numpy as np
import itertools
import src.world as wrld


def train_agent_with_value_iteration(agent, world, epochs=400, epsilon=1e-4, discount=0.9):
    values = np.zeros((world.height, world.width), np.float32)

    for i in range(epochs):
        maxValChange = 0
        for state in itertools.product(range(world.height), range(world.width)):
            localValChange  = values[state[0], state[1]]
            qVals           = np.zeros((4,), np.float32)
            for a in wrld.World.ACTIONS:
                newState, qVals[a]  = world.apply_action(a, np.array(state))
                qVals[a]            += discount * values[newState[0], newState[1]]
            values[state[0], state[1]]          = np.max(qVals)
            agent.policy[state[0], state[1]]    = np.argmax(qVals)

            localValChange  = np.abs(localValChange - values[state[0], state[1]])
            maxValChange    = np.max((maxValChange, localValChange))

        if (i + 1) % 50 == 0 or i == 0:
            print('[{}/{}] Max value change: {}'.format(i+1, epochs, maxValChange))
        if maxValChange < epsilon:
            print('[{}/{}] Max value change: {}'.format(i + 1, epochs, maxValChange))
            break


def train_agent_with_policy_iteration(agent, world, epochs=200, epsilon=1e-4, discount=0.98):
    policy = agent.policy.copy()

    while True:
        values = np.zeros((world.height, world.width), np.float32)

        currentValues = values.copy()
        for i in range(epochs):
            maxValChange    = 0

            for state in itertools.product(range(world.height), range(world.width)):
                localValChange  = currentValues[state[0], state[1]]
                newState, val   = world.apply_action(policy[state[0], state[1]], np.array(state))

                currentValues[state[0], state[1]] = val + discount * values[newState[0], newState[1]]

                localValChange  = np.abs(localValChange - currentValues[state[0], state[1]])
                maxValChange    = np.max((maxValChange, localValChange))

            values = currentValues.copy()
            if maxValChange < epsilon:
                break

        # Policy update
        newPolicy = policy.copy()
        for state in itertools.product(range(world.height), range(world.width)):
            qVals = np.zeros((4,), dtype=np.float32)
            for a in wrld.World.ACTIONS:
                newState, qVals[a]  = world.apply_action(a, np.array(state))
                qVals[a]            += discount * values[newState[0], newState[1]]
            newPolicy[state[0], state[1]] = np.argmax(qVals)

        if np.array_equal(policy, newPolicy):
            break
        else:
            policy = newPolicy.copy()

    agent.policy = policy

