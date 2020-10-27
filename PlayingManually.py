from PygamePlayCar import CarGame, PolicyFunction, DQN  # noqa: F401
# need to import PolicyFunction too, else python cannot find it (and also DQN)

g = CarGame()
g.play_game_manually()
