import itertools
import src.world as wrld

def plot(world, agent):
    import matplotlib
    matplotlib.use('TkAgg', force=True)
    import matplotlib.pyplot as plt
    import matplotlib.patches as ptchs

    offset  = 0.35
    diff    = 0.5
    arrowActionBaseOffset   = [(offset, 0), (0, -offset), (-offset, 0), (0 ,offset)]
    arrowActionsDiff        = [(-diff, 0), (0, diff), (diff, 0), (0, -diff)]

    fig, ax = plt.subplots()
    plt.hlines(range(0, world.height + 1), 0, world.width, colors='black')
    plt.vlines(range(0, world.width + 1), 0, world.height, colors='black')

    for state in itertools.product(range(world.height), range(world.width)):
        if world.worldMap[state[0], state[1]] == wrld.World.PENALTY:
            ax.add_patch(ptchs.Rectangle((state[1], state[0]), 1, 1, facecolor='black'))
        elif world.worldMap[state[0], state[1]] == wrld.World.REWARD:
            ax.add_patch(ptchs.Rectangle((state[1], state[0]), 1, 1, facecolor='yellow'))
        else:
            action = agent.policy[state[0], state[1]]
            plt.arrow(state[1] + 0.5 + arrowActionBaseOffset[action][0],
                      state[0] + 0.5 + arrowActionBaseOffset[action][1],
                      arrowActionsDiff[action][0], arrowActionsDiff[action][1],
                      width=0.05)

    plt.tight_layout()
    plt.show()
