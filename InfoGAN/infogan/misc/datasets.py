import numpy as np
from tensorflow.examples.tutorials import mnist
import os
import numpy as np
import pickle
import pdb

class Dataset(object):
    def __init__(self, images, labels=None):
        self._images = images.reshape(images.shape[0], -1)
        self._labels = labels
        self._epochs_completed = -1
        self._num_examples = images.shape[0]
        # shuffle on first run
        self._index_in_epoch = self._num_examples

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        print("NEXT BATCH========================")
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            if self._labels is not None:
                self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        if self._labels is None:
            return self._images[start:end], None
        else:
            return self._images[start:end], self._labels[start:end]


class MnistDataset(object):
    def __init__(self):
        data_directory = "MNIST"
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
        dataset = mnist.input_data.read_data_sets(data_directory)
        self.train = dataset.train
        # make sure that each type of digits have exactly 10 samples
        sup_images = []
        sup_labels = []
        rnd_state = np.random.get_state()
        np.random.seed(0)
        for cat in range(10):
            ids = np.where(self.train.labels == cat)[0]
            np.random.shuffle(ids)
            sup_images.extend(self.train.images[ids[:10]])
            sup_labels.extend(self.train.labels[ids[:10]])
        np.random.set_state(rnd_state)
        self.supervised_train = Dataset(
            np.asarray(sup_images),
            np.asarray(sup_labels),
        )
        self.test = dataset.test
        self.validation = dataset.validation
        self.image_dim = 28 * 28
        self.image_shape = (28, 28, 1)

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


class PolicyDataset(object):
    def __init__(self):
        # read in data
        forward_file_states = '/home/leegroup/Documents/gan_imitation/InfoGAN/data/forward_walker/expert_policies_states.npy'
        forward_file_actions = '/home/leegroup/Documents/gan_imitation/InfoGAN/data/forward_walker/expert_policies_actions.npy'
        
        backward_file_states = '/home/leegroup/Documents/gan_imitation/InfoGAN/data/backward_walker/expert_policies_states.npy'
        backward_file_actions = '/home/leegroup/Documents/gan_imitation/InfoGAN/data/backward_walker/expert_policies_actions.npy'

        backward_states = np.load(backward_file_states)[:, :-1]
        backward_actions = np.load(backward_file_actions)

        forward_states = np.load(forward_file_states)[:, :-1]
        forward_actions = np.load(forward_file_actions)

        # states = np.concatenate((backward_states, forward_states),axis = 0)
        # actions = np.concatenate((backward_actions, forward_actions),axis = 0)

        states = np.concatenate((forward_states, backward_states),axis = 0)
        actions = np.concatenate((forward_actions, backward_actions),axis = 0)


        self.data = {'states': states, 'actions': actions}
        self.state_dim = 17*1
        self.state_shape = (17,1,1)

        self.action_dim = 6*1
        self.action_shape = (6,1,1)

    def next_batch(self, batch_size):
        n = self.data['states'].shape[0]

        rand_idx = np.random.choice(n, size=batch_size)

        
        state_noise = np.random.normal(0, 0.04, (batch_size,17))
        action_noise = np.random.normal(0, 0.04, (batch_size,6))
        rand_states = self.data['states'][rand_idx, :] + state_noise
        rand_actions = self.data['actions'][rand_idx, :] + action_noise

        
        return rand_states, rand_actions
    



