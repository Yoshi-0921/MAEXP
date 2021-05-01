

world = World()
num_agents = config.num_agents
num_landmarks = config.num_landmarks
world.agents = [Agent() for i in range(num_agents)]
for i, agent in enumerate(world.agents):
    agent.name = f"agent {i}"
    agent.collide = True
world.landmarks = [Landmark() for i in range(num_landmarks)]
for i, landmark in enumerate(world.landmarks):
    landmark.name = f"landmark {i}"
    landmark.collide = False
world.map = Exp6_Map(config)

return world