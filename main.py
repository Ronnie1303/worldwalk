import src.world as world
import src.agent as agent
import src.train as training
import src.plottingTools as plttools


if __name__ == '__main__':
    w = world.World()
    a = agent.AgentWithPolicy(w.height, w.width)

    training.train_agent_with_value_iteration(a, w)
    plttools.plot(w, a)



