import collections
import random
import time

import cv2
import numpy as np

import settings

snakeSize = 1
WIN_NAME = settings.WIN_NAME
pixGAP = settings.GAP

white = (255, 255, 255)
orange = (51, 173, 255)


def im_write(img, text, coords):
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_color = (255, 255, 255)
    line_size = 1
    line_type = cv2.LINE_AA
    cv2.putText(img, str(text),
                coords, font, line_size, line_color, lineType=line_type)
    return img


def draw_square(img, x, y, l, c, pix, sepPixels, world):
    maxNum = world.shape[0]
    if x >= maxNum:
        return img
    if y >= maxNum:
        return img
    if x < 0:
        return img
    if y < 0:
        return img
    x_pix = x * pix
    y_pix = y * pix
    pt1 = [x_pix, y_pix]
    pt2 = [x_pix + l, y_pix + l]
    if sepPixels:
        if world[x][y][0] == 1 or world[x][y][0] == 2:
            if x - 1 > 0:
                if world[x - 1][y][0] == 1 or world[x - 1][y][0] == 2:
                    if not (world[x - 1][y][1] == world[x][y][1] - 1 or
                            world[x - 1][y][1] == world[x][y][1] + 1):
                        pt1[0] += pixGAP
                else:
                    pt1[0] += pixGAP

            if x + 1 < maxNum:
                if world[x + 1][y][0] == 1 or world[x + 1][y][0] == 2:
                    if not (world[x + 1][y][1] == world[x][y][1] - 1 or
                            world[x + 1][y][1] == world[x][y][1] + 1):
                        pt2[0] -= pixGAP
                else:
                    pt2[0] -= pixGAP

            if y - 1 > 0:
                if world[x][y - 1][0] == 1 or world[x][y - 1][0] == 2:
                    if not (world[x][y - 1][1] == world[x][y][1] - 1 or
                            world[x][y - 1][1] == world[x][y][1] + 1):
                        pt1[1] += pixGAP
                else:
                    pt1[1] += pixGAP

            if y + 1 < maxNum:
                if world[x][y + 1][0] == 1 or world[x][y + 1][0] == 2:
                    if not (world[x][y + 1][1] == world[x][y][1] - 1 or
                            world[x][y + 1][1] == world[x][y][1] + 1):
                        pt2[1] -= pixGAP
                else:
                    pt2[1] -= pixGAP

    pt1 = tuple(pt1)
    pt2 = tuple(pt2)
    cv2.rectangle(img, pt1, pt2, c, thickness=-1, lineType=cv2.LINE_AA)
    return img


def renderWorld(world_info, pixels, sepPixels):
    world = world_info["world"]
    snake_body = world_info["snake_body"]
    food = world_info["food"]
    head = world_info["head"]
    display = np.zeros([world.shape[0] * pixels, world.shape[1] * pixels, 3], dtype=np.uint8)
    for x, y in snake_body:
        draw_square(display, x, y, pixels, settings.COLORS[settings.WORLD_INFO['body']], pixels, sepPixels, world)
    draw_square(display, food[0], food[1], pixels, settings.COLORS[settings.WORLD_INFO['food']], pixels, sepPixels, world)
    draw_square(display, head[0], head[1], pixels, settings.COLORS[settings.WORLD_INFO['head']], pixels, sepPixels, world)

    display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
    return display


class SnakeEnv:
    def __init__(self):
        self.action_space = collections.namedtuple('n', 'x')
        self.action_space.n = 4
        # DQN DATA
        self.ACTION_SPACE_SIZE = 4

        self.reset()

    def reset(self):
        cv2.namedWindow(WIN_NAME)
        cv2.moveWindow(WIN_NAME, -16, 0)
        self.cur_step = 0
        self.show_info = {}

        self.st = time.time()

        self.snakeList = []
        self.snakeLen = 3
        self.x1_change = 0
        self.y1_change = 0

        self.x1 = settings.world_size // 2
        self.y1 = settings.world_size // 2

        self.snakeHead = [self.x1, self.y1]

        self.world = []

        self.foodx = round(random.randrange(0, settings.world_size - snakeSize) / snakeSize) * snakeSize
        self.foody = round(random.randrange(0, settings.world_size - snakeSize) / snakeSize) * snakeSize

        self.wasFoodCollected = False
        self.done = False
        self.state = []
        self.reward = 0

        self.foodCollected = 0
        self.timed_steps = 0
        self.genWorldInfo()
        self.getState()
        return self.ret_state

    def step(self, action, info=None):
        self.wasFoodCollected = False
        self.st = time.time()
        self.cur_step += 1

        action = int(action)
        if info is None:
            info = {}
        self.show_info = info
        self.move(action)
        self.calculate_reward()
        self.calculate_done()
        self.genWorldInfo()
        self.getState()
        return self.ret_state, self.reward, self.done, self.wasFoodCollected

    def move(self, action):
        if action == 0:
            self.x1_change = -snakeSize
            self.y1_change = 0
        elif action == 1:
            self.x1_change = snakeSize
            self.y1_change = 0
        elif action == 2:
            self.x1_change = 0
            self.y1_change = -snakeSize
        elif action == 3:
            self.x1_change = 0
            self.y1_change = snakeSize
        else:
            print("ERR")

        self.x1 += self.x1_change
        self.y1 += self.y1_change

        self.snakeHead = []
        self.snakeHead.append(self.x1)
        self.snakeHead.append(self.y1)
        self.snakeList.append(self.snakeHead)

        if len(self.snakeList) > self.snakeLen:
            del self.snakeList[0]

    def calculate_reward(self):
        self.reward = 0
        if self.foodCollected >= settings.FOOD_FOR_REWARD:
            self.reward += settings.REWARD_PER_FRAME
        else:
            self.reward += settings.PENALTY_PER_FRAME

        if self.x1 == self.foodx and self.y1 == self.foody:
            while True:
                self.foodx = round(random.randrange(0, settings.world_size - snakeSize) / snakeSize) * snakeSize
                self.foody = round(random.randrange(0, settings.world_size - snakeSize) / snakeSize) * snakeSize
                good = True
                for coord in self.snakeList:
                    if self.foodx == coord[0] and self.foody == coord[1]:
                        good = False
                        break

                if good:
                    break
            self.timed_steps = 0
            self.wasFoodCollected = True
            self.snakeLen += 1
            self.foodCollected += 1
            self.reward += settings.FOOD_REWARD
        if self.snakeLen == settings.world_size * settings.world_size:
            self.reward += settings.REWARD_FOR_WIN

    def calculate_done(self):
        self.timed_steps += 1

        if self.timed_steps > settings.TIMEOUT_STEPS and self.foodCollected < settings.foodForTimeout:
            self.done = True
            self.reward += settings.TIMEOUT_PENALTY

        if self.x1 >= settings.world_size or self.x1 < 0 or self.y1 >= settings.world_size or self.y1 < 0:
            self.done = True
            self.reward += settings.CRASH_PENALTY

        for x in self.snakeList[:-1]:
            if x == self.snakeHead:
                self.done = True
                self.reward += settings.CRASH_PENALTY
                break
        if self.snakeLen == settings.world_size * settings.world_size:
            self.done = True

    def genWorldInfo(self):
        self.world = np.zeros((settings.world_size, settings.world_size, 2), dtype=np.uint8)
        num = 0
        for x, y in reversed(self.snakeList):
            try:
                self.world[x][y] = [settings.WORLD_INFO["body"], num]
            except IndexError:
                pass
            num += 1
        try:
            self.world[self.snakeHead[0]][self.snakeHead[1]] = [settings.WORLD_INFO["head"], 0]
        except IndexError:
            pass
        try:
            self.world[self.foodx][self.foody] = [settings.WORLD_INFO["food"], 10000]
        except IndexError:
            pass
        self.world_info = {'head': self.snakeHead,
                           'snake_body': self.snakeList[:-1],
                           "world": self.world,
                           'food': [self.foodx, self.foody]}

    def getState(self):
        if settings.USE_PIXELS:
            state = renderWorld(self.world_info, settings.state_pixels_per_unit, False)
            self.state = state
            if settings.WINDOW_VIEW_MODE:
                self.ret_state = []  # self.getWindow()
            else:
                self.ret_state = self.state[:]
        else:
            self.state = []
            self.ret_state = []

    def render(self, sleep=True, fps=settings.FPS):
        display = renderWorld(self.world_info, settings.pixels_per_unit, True)
        for key in self.show_info:
            display = im_write(display, f"{key}: {self.show_info[key][0]}",
                               self.show_info[key][1])
        cv2.imshow(WIN_NAME, display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
        if sleep:
            et = time.time()
            sleeptime = (1 / fps) - (et - self.st)
            if sleeptime < 0:
                sleeptime = 0
            time.sleep(sleeptime)


if __name__ == '__main__':
    envf = SnakeEnv()
    envf.reset()
    envf.step(1)
    envf.render()
