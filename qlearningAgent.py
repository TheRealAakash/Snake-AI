import random
from collections import deque

import numpy as np

import settings
from get_keys import key_check
from models import main_model, choose_model
from tensorboard_modded import ModifiedTensorBoard

USE_CHECKPOINT = False

DISCOUNT = settings.DISCOUNT
REPLAY_MEMORY_SIZE = settings.REPLAY_MEMORY_SIZE  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = settings.MIN_REPLAY_MEMORY_SIZE  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = settings.MINIBATCH_SIZE  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = settings.UPDATE_TARGET_EVERY  # Terminal states (end of episodes)
MODEL_NAME = settings.MODEL_NAME
NORMALIZE_BY = settings.NORMALIZE_BY

MEMORY_FRACTION = settings.MEMORY_FRACTION  # 0.20

LEARNING_RATE = settings.LEARNING_RATE

# For more repetitive results
random.seed(1)
np.random.seed(1)

# SAVE / LOAD
LOAD_PREV_MODEL = settings.LOAD_PREV_MODEL

key_check()
keys = []


def extend_image(source, image):
    source = list(source)
    image = list(image)
    source.extend(image)
    source = np.array(source)
    return source


class DQNAgent:
    def __init__(self, num_actions, sample_state):
        self.memory = deque([np.zeros(sample_state.shape) for _ in range(settings.NUM_STACKED)],
                            maxlen=settings.NUM_STACKED)
        self.sample_state = sample_state

        # Main model
        if LOAD_PREV_MODEL:
            mdl_file = choose_model()
            print(f"Using model: {mdl_file}")
            self.model = self.create_model(num_actions, sample_state)
            self.model.load_weights(mdl_file)

        else:
            self.model = self.create_model(num_actions, sample_state)

        # Target network
        self.target_model = self.create_model(num_actions, sample_state)
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}")

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def reset(self):
        self.memory = deque([np.zeros(self.sample_state.shape) for _ in range(settings.NUM_STACKED)],
                            maxlen=settings.NUM_STACKED)

    def create_model(self, num_actions, sample_state):
        sample_state = self.process(sample_state)
        model = main_model(num_actions, sample_state)
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state):
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch]) / NORMALIZE_BY
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch]) / NORMALIZE_BY
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X) / NORMALIZE_BY, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False,
                       callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        state = np.array(state).reshape(-1, *state.shape) / NORMALIZE_BY
        return self.model.predict(state)[0]

    # noinspection PyMethodMayBeStatic

    def process(self, state):
        new_state = []
        self.memory.append(state)
        if settings.USE_STACK:
            for i in range(len(self.memory)):
                new_state = extend_image(new_state, self.memory[i])
        else:
            new_state = state
        return new_state

    def step(self, state):
        state = self.process(state)
        return state
