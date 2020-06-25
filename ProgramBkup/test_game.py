import random
import time
from statistics import mean

import get_keys
import settings
from snake import SnakeEnv

prev_act = 0
random.seed(0)


def get_input():
    global prev_act
    keys = get_keys.key_check()
    action = prev_act
    if "A" in keys:
        action = 3
    elif "D" in keys:
        action = 1
    elif "W" in keys:
        action = 0
    elif "S" in keys:
        action = 2
    prev_act = action
    return action


def main():
    times = []
    env = SnakeEnv()
    for i in range(100):
        st = time.time()
        done = False
        env.reset()
        score = 0
        food = 0
        while not done:
            info = {"Food": (food, (10, 30))}
            state, reward, done = env.step(get_input(), info=info)
            score += reward
            if reward == settings.FOOD_REWARD:
                food += 1

            env.render(sleep=False)
            # print(reward)
            if done:
                et = time.time()
                times.append(et - st)
                # quit()
                break
    print(1 / (mean(times)), end=" games per second\n")
    print(1 / (max(times)), end=" slowest games per second\n")
    print(1 / (min(times)), end=" fastest games per second\n")


main()
