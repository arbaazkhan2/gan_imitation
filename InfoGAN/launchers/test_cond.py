import tensorflow as tf
import gym
from infogan.misc.datasets import MnistDataset, PolicyDataset
import argparse
import pdb

def data_test():
	dataset = PolicyDataset()

	env = gym.make('Walker2d-v1')
	obs = env.reset()

	states = dataset.data['states']
	actions = dataset.data['actions']


	for i in range(10000):
		env.render()

		action = actions[i, :]


		obs, reward, done, _ = env.step(action)


		# if done:
		# 	break


def test():
	pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('test stuff'))
    parser.add_argument('-t', '--test', type=bool,
                        help='set to true to save the data (states, actions, rewards, next_states) that the policy generates. and not train the model',
                        default=False)
    parser.add_argument('-d', '--data', type=bool,
    					help='',
    					default=False)


    args = parser.parse_args()
    if args.data:
    	data_test()


    if args.test:
    	pass


