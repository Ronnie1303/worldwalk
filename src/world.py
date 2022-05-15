import numpy as np

def distance_mahalanobis(posA, posB):
    return np.abs(posA[0] - posB[0]) + np.abs(posA[1] - posB[1])


def distance_euclid(posA, posB):
    return np.sqrt(np.square(posA[0] - posB[0]) + np.square(posA[1] - posB[1]))


class World:
    # Coeffs
    REWARD          = 100
    PENALTY         = -1000
    WALL_PENALTY    = -5
    WALK_PENALTY    = -1

    # Actions
    MOVE_LEFT   = 0
    MOVE_UP     = 1
    MOVE_RIGHT  = 2
    MOVE_DOWN   = 3
    ACTIONS     = [MOVE_LEFT, MOVE_UP, MOVE_RIGHT, MOVE_DOWN]

    def __init__(self, seed=0, distanceMetric=distance_mahalanobis):
        # Hardcoded dims
        self.height = 10
        self.width  = 20

        # Set distance metric
        self.distanceMetric = distanceMetric

        # World map is NxM array of ints, 100 indicates target and -100 indicates death
        self.worldMap   = np.zeros((self.height, self.width), dtype=int)
        self.targetLoc  = np.zeros((2,), dtype=int)

        self.generate_world(seed)


    def __str__(self):
        worldRepresentation = ''
        for i in range(self.height - 1, -1, -1):
            worldRepresentation += '|'
            for j in range(self.width):
                if self.worldMap[i,j] == 0:
                    worldRepresentation += ' |'
                elif self.worldMap[i, j] == World.REWARD:
                    worldRepresentation += '*|'
                else:
                    worldRepresentation += 'X|'
            worldRepresentation += '\n'
        return worldRepresentation


    def generate_world(self, seed: int):
        if seed == 0:
            self.worldMap[9, 17]        = World.REWARD
            self.worldMap[2:5, 1]       = World.PENALTY
            self.worldMap[1:5, 2:4]     = World.PENALTY
            self.worldMap[2:6, 3:7]     = World.PENALTY
            self.worldMap[2:6, 9:14]    = World.PENALTY
            self.worldMap[6:7, 11:14]   = World.PENALTY
            self.worldMap[4:6, 14:16]   = World.PENALTY
            self.worldMap[3, 14:18]     = World.PENALTY
            self.worldMap[0:3, 14:18]   = World.PENALTY
            self.worldMap[-3:, 0:5]     = World.PENALTY
            self.worldMap[-2:, 0:7]     = World.PENALTY
            self.worldMap[-2:, 11:14]   = World.PENALTY
            self.worldMap[-4:-2, 17:]   = World.PENALTY
            self.worldMap[-2:, -1]      = World.PENALTY
        else:
            raise Exception('Generating world with nonzero seeds not implemented yet!')


    def apply_action(self, action: int, state: np.ndarray) -> 'tuple':
        if not 0 <= action <= 3:
            raise Exception('[ERROR] Action ID ({}) not in valid range! [0,3]'.format(action))

        stateUpdates    = [[0, -1], [1, 0], [0, 1], [-1, 0]]
        newState        = state + stateUpdates[action]

        if not 0 <= newState[0] < self.height or not 0 <= newState[1] < self.width:
            # Implicates attempt to go out of bounds
            return state, World.WALL_PENALTY
        else:
            distanceDelta   = self.distanceMetric(state, self.targetLoc) - self.distanceMetric(newState, self.targetLoc)
            state           = newState

        if self.worldMap[newState[0], newState[1]] == World.PENALTY:
            reward = World.PENALTY
        else:
            reward = World.WALK_PENALTY + distanceDelta + self.worldMap[state[0], state[1]]

        return state, reward

