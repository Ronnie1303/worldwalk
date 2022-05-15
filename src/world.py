import numpy as np
import itertools


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

    def __init__(self, seed=0, height=10, width=20, distanceMetric=distance_mahalanobis):
        if height < 5 or width < 5:
            raise Exception('Width and height of world map must be at least 5!')

        if seed == 0:
            self.height = 10
            self.width  = 20
        else:
            self.height = height
            self.width  = width

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
            # Cellular automaton
            # 1. Initialize world according to seed
            pattern         = [int(x) for x in str(seed)]
            numElements     = self.worldMap.size
            #flatWorld       = np.tile(pattern, int(np.ceil(numElements / len(pattern))))
            #flatWorld       = flatWorld[:numElements]

            rng                         = np.random.default_rng(seed)
            flatWorld                   = rng.integers(low=0, high=10, size=numElements)
            flatWorld[flatWorld < 5]    = 0
            flatWorld[flatWorld >= 5]   = World.PENALTY
            self.worldMap               = np.reshape(flatWorld, (self.height, self.width))

            # 2. Iterate map
            for k in range(5):
                worldCopy = self.worldMap.copy()
                for pos in itertools.product(range(self.height), range(self.width)):
                    wallCount = 0

                    for i in range(-1,2):
                        for j in range(-1, 2):
                            if not 0 <= pos[0] + i < self.height or not 0 <= pos[1] + j < self.width:
                                wallCount += 1
                            elif self.worldMap[pos[0] + i, pos[1] + j] == World.PENALTY:
                                wallCount += 1

                    if wallCount < 4:
                        worldCopy[pos[0], pos[1]] = 0
                    elif wallCount > 4:
                        worldCopy[pos[0], pos[1]] = World.PENALTY
                    else:
                        pass

                self.worldMap = worldCopy.copy()

            # 2.5 Invert map
            self.worldMap[worldCopy == 0]               = World.PENALTY
            self.worldMap[worldCopy == World.PENALTY]   = 0

            # 3. Generate target location
            sel = pattern[0]
            countedValidTiles = 0
            for i in range(self.height - 1, -1, -1):
                for j in range(self.width - 1, -1, -1):
                    if self.worldMap[i,j] == 0:
                        countedValidTiles += 1

                    if countedValidTiles == sel:
                        self.worldMap[i, j] = World.REWARD
                        self.targetLoc[:]   = [i ,j]
                        break
                if self.worldMap[i, j] == World.REWARD:
                    break


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

