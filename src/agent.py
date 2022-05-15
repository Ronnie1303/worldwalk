import numpy as np
import itertools


class Agent:
    def __init__(self):
        # State is triplet [x, y, status]; x and y indicate position; status indicates if agent alive ([1,-1])
        self.state = np.array([0, 0, 1], dtype=int)

    def action(self, worldState):
        return 0

    def reset_state(self):
        self.state = np.array([0, 0, 1], dtype=int)


class AgentWithPolicy(Agent):
    def __init__(self, worldHeight, worldWidth):
        super().__init__()
        self.policy = np.zeros((worldHeight, worldWidth), dtype=int)


    def action(self, worldState):
        return self.policy[worldState[0], worldState[1]]


class AgentWithNeuralNet(Agent):
    def __init__(self, neuralNetwork):
        super().__init__()
        self.neuralNet = neuralNetworki


    def action(self, worldState):
        return self.neuralNet(worldState)