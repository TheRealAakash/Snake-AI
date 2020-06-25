import os
import random
import time
import tkinter as tk
from threading import Thread

import cv2
# import h5py
import numpy as np
import tensorflow as tf

# from modifiedtb import ModifiedTensorBoard
import SnakeEnv
import helper
import settings
from cv2_viewer import show_video
from get_keys import KeyChecker
from qlearningAgent import DQNAgent

os.mkdir(f"past_games\\{settings.MODEL_NAME}")
os.mkdir(f"models\\{settings.MODEL_NAME}")

keyChecker = KeyChecker()

MODEL_NAME = settings.MODEL_NAME
MIN_REWARD = settings.MIN_REWARD  # For model save
POINTS_PER_FOOD = settings.FOOD_REWARD

# EXPLORE VS EXPLOIT
MIN_EPSILON = settings.MIN_EPSILON
EXPLORE_EVERY = settings.EXPLORE_EVERY
EXPLOIT_EVERY = settings.EXPLOIT_EVERY
EXPLORE_RANDOM_RATE = settings.EXPLORE_RANDOM_RATE

#  Stats settings
AGGREGATE_STATS_EVERY = settings.AGGREGATE_STATS_EVERY  # episodes
UPDATE_STATS_EVERY = settings.UPDATE_STATS_EVERY  # episodes
SAVE_EVERY = settings.SAVE_EVERY
SHOW_PREVIEW = settings.SHOW_PREVIEW
ep_rewards = [settings.CRASH_PENALTY]

env = SnakeEnv.SnakeEnv()

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.set_random_seed(1)
RANDOM_MODE = settings.RANDOM_MODE
# SAVE / LOAD
unit_to_find = settings.unit_to_find
GAME_SAVE = settings.GAME_SAVE

USE_MODEL = False
SHOW_SENSORS = False
EXPAND_SENSOR = False

# Memory fraction, used mostly when training multiple agents
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
# backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')

if not os.path.isdir(GAME_SAVE):
    os.makedirs(GAME_SAVE)

if not os.path.isdir('logs'):
    os.makedirs('logs')

EPISODES = settings.EPISODES
epsilon = settings.epsilon  # not a constant, going to be decayed

START_EPSILON = settings.START_EPSILON
START_EPSILON_DECAYING = settings.START_EPSILON_DECAYING
END_EPSILON_DECAYING = settings.END_EPSILON_DECAYING
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)


def save_video(frames, name, ep_rew):
    save = [frames, f"End Food: {ep_rew}"]
    np.save(f"{GAME_SAVE}/{name}", save)  # FIX


def create_tk():
    root = tk.Tk()
    root.title("Stats")
    w = root.winfo_screenwidth()
    h = 180
    x = -9
    y = root.winfo_screenheight() - h - 59
    root.geometry('%dx%d+%d+%d' % (w, h, x, y))
    root.protocol("WM_DELETE_WINDOW", root.iconify)
    label = tk.Text(root, fg="snow", bg="black")
    label.config(font=("Courier", 20))
    label.pack(expand='YES', fill='both')
    label.update()
    return root, label


def user_input():
    global SHOW_SENSORS
    global SHOW_PREVIEW
    global USE_MODEL
    global EXPAND_SENSOR
    HIST_SHOWING = False
    hist = None
    while True:
        if keyChecker.checkKey("P"):
            if SHOW_PREVIEW:
                SHOW_PREVIEW = False
                print("Pausing Preview...")
            else:
                SHOW_PREVIEW = True
                print("Un-pausing Preview...")

        if keyChecker.checkKey("E"):
            if USE_MODEL:
                print("Exploring...")
                USE_MODEL = False
            else:
                print("Exploiting...")
                USE_MODEL = True

        if keyChecker.checkKey("R"):
            if EXPAND_SENSOR:
                print("Expanding Sensor...")
                EXPAND_SENSOR = False
            else:
                print("De-Expanding Sensor...")
                EXPAND_SENSOR = True

        if keyChecker.checkKey("S"):
            if SHOW_SENSORS:
                print("Hiding Sensors")
                settings.SHOW_SENSORS = False
                SHOW_SENSORS = False
                cv2.destroyWindow('Snake View')
            else:
                print("Showing Sensors")
                settings.SHOW_SENSORS = True
                SHOW_SENSORS = True

        if keyChecker.checkKey("H"):
            if not HIST_SHOWING:
                HIST_SHOWING = True
                HIST_SHOWING = True
                hist = Thread(target=show_video, daemon=True)
                hist.start()
                print("Showing hist...")

        if HIST_SHOWING:
            if not hist.is_alive():
                HIST_SHOWING = False
                print("History ended...")


def extend_image(source, image):
    source = list(source)
    image = list(image)
    source.extend(image)
    source = np.array(source)
    return source


agent = DQNAgent(env.ACTION_SPACE_SIZE, env.reset())


def main():
    # noinspection PyGlobalUndefined
    global SHOW_PREVIEW, max_reward, average_reward, min_reward, max_food_100, num_food_avg, max_rendered
    global agent
    global epsilon
    global keys
    # Iterate over episodes
    root, label = create_tk()
    start_time = time.time()

    t = range(1, EPISODES + 1)

    Thread(target=user_input, daemon=True).start()

    food_total = []

    steps_total = []
    best_ep_rewards = 0
    wins = 0
    streak = 0
    max_rendered = 0
    for episode in t:
        game_video = []
        # Update tensorboard step every episode
        agent.tensorboard.step = episode

        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        step = 1

        # Reset environment and get initial state
        # Reset flag and start iterating until episode ends
        SHOW_CURRENT = False
        done = False
        num_food = 0
        reward = 0
        current_state = env.reset()
        agent.reset()
        current_state = agent.step(current_state)
        while not done:
            if RANDOM_MODE.lower() == 'np':
                random_val = np.random.random()
            else:
                random_val = random.random()
            # This part stays mostly the same, the change is to query a model for Q values
            if (random_val > epsilon or episode <= START_EPSILON or not (episode - 1) % EXPLOIT_EVERY) and not (
                    not (episode - 2) % EXPLORE_EVERY and random_val > EXPLORE_RANDOM_RATE) or USE_MODEL:
                # Get action from Q table
                action = np.argmax(agent.get_qs(current_state))
                mode = "EXPLOIT"
            else:
                # Get random action
                if RANDOM_MODE.lower() == 'np':
                    action = np.random.randint(0, env.ACTION_SPACE_SIZE)
                else:
                    action = random.randint(0, env.ACTION_SPACE_SIZE - 1)
                mode = "EXPLORE"
            info = {
                'Step': (step, (10, 30)),
                'Food': (num_food, (1000, 30)),
                'Episode': (episode, (10, 1190)),
                'Score': (round(episode_reward, 2), (10, 60)),
                'Mode': (mode, (500, 1190)),
                }
            new_state, reward, done, wasFoodCollected = env.step(action, info=info)
            new_state = agent.step(new_state)
            # Transform new continuous state to new discrete state and count reward
            episode_reward += reward
            if wasFoodCollected:
                num_food += 1
            if SHOW_PREVIEW or episode == 1 or (num_food >= max(food_total) and num_food != 0) or (
                    num_food > max_rendered and num_food != 0):
                max_rendered = num_food
                # SHOW_CURRENT = True
                if not done:
                    env.render(False)
            if SHOW_SENSORS:
                current_state_show = current_state[:]
                if EXPAND_SENSOR:
                    current_state_show = cv2.resize(current_state_show, (0, 0), fx=settings.EXPAND_RATIO, fy=settings.EXPAND_RATIO, interpolation=cv2.INTER_LINEAR)
                cv2.imshow("Snake View", current_state_show)
                cv2.waitKey(1)

            # Every step we update replay memory and train main network
            agent.update_replay_memory((current_state, action, reward, new_state, done))
            agent.train(done)

            current_state = new_state
            game_video.append([env.world_info, info])
            step += 1

        if reward == settings.REWARD_FOR_WIN:
            wins += 1
            streak += 1
        else:
            streak = 0
        # END OF EPISODE
        if episode == 1:
            best_ep_rewards = episode_reward
        else:
            if episode_reward > best_ep_rewards:
                best_ep_rewards = episode_reward
        if food_total:
            max_food = max(food_total)
        else:
            max_food = 0
        food_total.append(num_food)
        ep_rewards.append(episode_reward)
        steps_total.append(step)

        # Append episode reward to a list and log stats (every given number of episodes)
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        num_food_avg = sum(food_total[-AGGREGATE_STATS_EVERY:]) / len(food_total[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_food_100 = max(food_total[-AGGREGATE_STATS_EVERY:])
        avg_steps = sum(steps_total[-AGGREGATE_STATS_EVERY:]) / len(steps_total[-AGGREGATE_STATS_EVERY:])
        name = f"name{MODEL_NAME}_episode{episode}_max{max_reward:_>7.2f}_avg{average_reward:_>7.2f}_min{min_reward:_>7.2f}_food{max_food_100}_avgfood{num_food_avg}_time{int(time.time())}"

        if not episode % UPDATE_STATS_EVERY or episode == 1:
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
                                           epsilon=epsilon, num_food_avg=num_food_avg,
                                           food_max=max_food_100, avg_steps=avg_steps, step=episode)

            # Save model, but only when min reward is greater or equal a set value
            if min_reward >= MIN_REWARD:
                agent.model.save(f"""models/{settings.MODEL_NAME}/{name}.model""")
        if episode_reward > best_ep_rewards or episode == 1 or num_food > max_food:
            best_ep_rewards = episode_reward
            agent.model.save(f"""models/{settings.MODEL_NAME}/{name}.model""")
            save_video(game_video, f"food_{num_food}rew_{episode_reward}episode_{episode}model_{MODEL_NAME}", num_food)

        if not episode % SAVE_EVERY:
            # noinspection PyUnboundLocalVariable
            agent.model.save(f"""models/{settings.MODEL_NAME}/{name}.model""")
        time_data = helper.get_time_data(start_time, EPISODES, episode, "Episodes")

        # noinspection PyUnboundLocalVariable
        # {episode:>5d}
        text = f'Episode: {episode}/{EPISODES}, ' \
               f'Wins: {wins}, ' \
               f'Win Streaks: {streak}, ' \
               f'food coll: {num_food}, ' \
               f'avg food coll: {num_food_avg:>.2f}, ' \
               f'max recent food coll: {max_food_100} {food_total[-AGGREGATE_STATS_EVERY:].count(max_food_100)}, ' \
               f'max food coll: {max(food_total)} {food_total.count(max(food_total))}, ' \
               f'average reward: {average_reward:>.1f}, ' \
               f'average game len: {avg_steps:>.2f} ' \
               f'current epsilon: {epsilon:>.2f}, ' \
               f'time data: {time_data}, {round((episode / EPISODES) * 100, 2)}%            '

        label.insert(tk.END, text)
        label.update()
        label.delete(1.0, tk.END)
        # helper.update_print_line(text)
        # Decay epsilon
        if epsilon > MIN_EPSILON:
            if START_EPSILON_DECAYING:
                epsilon -= epsilon_decay_value
                epsilon = max(MIN_EPSILON, epsilon)


if __name__ == '__main__':
    main()
