from algo.rdpg import Rdpg

if __name__ == '__main__':
    agent = Rdpg((84, 84, 3), 2)

    # agent.exploration_learn(5000)
    agent.train(5000)
    agent.test(15)
