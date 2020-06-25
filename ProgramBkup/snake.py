import collections
import time

import cv2
import gym
import numpy as np

import settings

pixels_per_unit = settings.pixels_per_unit
world_size = settings.world_size
state_pixels_per_unit = settings.state_pixels_per_unit

black = (0, 0, 0)
white = (255, 255, 255)
red = (0, 0, 255)
cyan = (255, 255, 0)
blue = (255, 0, 0)
green = (0, 255, 0)
gray = (169, 169, 169)
orange = (51, 173, 255)
light_blue = (255, 191, 0)
FPS = settings.FPS

WIN_NAME = settings.WIN_NAME

USE_PIXELS = settings.USE_PIXELS


def im_write(img, text, coords):
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_color = (255, 255, 255)
    line_size = 1
    line_type = cv2.LINE_AA
    cv2.putText(img, str(text),
                coords, font, line_size, line_color, lineType=line_type)
    return img


def draw_square(img, x, y, l, c, pix):
    x *= pix
    y *= pix
    pt1 = (x, y)
    pt2 = (x + l, y + l)
    cv2.rectangle(img, pt1, pt2, c, thickness=-1, lineType=cv2.LINE_AA)
    return img


def expandImg(img, pixels=pixels_per_unit):
    size = img.shape[0] * pixels
    display = np.zeros((size, size, 3), np.uint8)
    # void, food, head, body, wall
    # cvtr = [black, red, green, white, orange]
    y = 0
    for col in img:
        x = 0
        for val in col:
            color = tuple(map(int, val))
            draw_square(display, x, y, pixels // state_pixels_per_unit, color, pixels)
            x += 1
        y += 1
    return display


# noinspection PyAttributeOutsideInit
class SnakeEnv:
    def __init__(self):
        self.env = gym.make('snake-v0')
        self.env.reset()
        self.env.__init__(grid_size=[world_size, world_size], unit_size=state_pixels_per_unit,
                          unit_gap=settings.STATE_GAP, random_init=False)
        self.env.seed(1)
        self.game_controller = self.env.controller
        self.grid_object = self.game_controller.grid

        self.action_space = collections.namedtuple('n', 'x')
        self.action_space.n = 4
        # DQN DATA
        self.ACTION_SPACE_SIZE = 4
        self.reward_correct = {
            -1: settings.CRASH_PENALTY,
            0: settings.PENALTY_PER_FRAME,
            1: settings.FOOD_REWARD,
            }

    def reset(self):
        self.st = time.time()
        cv2.namedWindow(WIN_NAME)
        cv2.moveWindow(WIN_NAME, -16, 0)
        self.show_info = {}
        self.state = self.env.reset()
        self.getState()
        # self.getGameDisplay()
        self.food_coll = 0
        self.cur_step = 0
        self.timed_steps = 0
        self.snake_x = 0
        self.snake_y = 0
        return self.ret_state

    def step(self, action, info=None):
        self.st = time.time()
        self.cur_step += 1

        action = int(action)
        if info is None:
            info = {}
        self.show_info = info
        self.state, self.reward, self.done, _ = self.env.step(action)
        self.getState()
        # self.getGameDisplay()

        self.isFinished()

        self.calculate_reward()
        self.calculate_done()
        return self.state, self.reward, self.done

    def isFinished(self):
        if self.food_coll < world_size * world_size - 10:
            self.finished = False
        else:
            self.finished = True
            empty = 0
            for col in self.state:
                for item in col:
                    if item == self.grid_object.SPACE_COLOR:
                        empty += 1
            if empty < 10:
                self.finished = False

    def calculate_reward(self):
        self.reward = self.reward_correct[self.reward]
        if self.reward == settings.FOOD_REWARD:
            self.food_coll += 1
            self.timed_steps = 0
        elif self.food_coll >= settings.FOOD_FOR_REWARD:
            self.reward_correct[0] = settings.REWARD_PER_FRAME
        elif self.finished:
            self.reward = settings.REWARD_FOR_WIN
        else:
            self.reward_correct[0] = settings.PENALTY_PER_FRAME

    def calculate_done(self):
        self.timed_steps += 1
        if self.timed_steps > settings.TIMEOUT_STEPS:
            self.done = True
        if self.finished:
            self.done = True

    def write_info(self):
        for key in self.show_info:
            self.display = im_write(self.display, f"{key}: {self.show_info[key][0]}",
                                    self.show_info[key][1])

    def getDisplay(self):
        # void, food, head, body, wall
        # cvtr = [black, red, green, white]
        display = expandImg(self.stateExp)
        return display

    def getWindow(self):
        self.getHead()
        # void, food, head, body, wall
        size = settings.VIEW_WINDOW_SIZE
        size_dir = size // 2
        world_center = (size // 2, size // 2)
        top_corner = (self.snake_x - size_dir, self.snake_y - size_dir)
        btm_corner = (self.snake_x + size_dir, self.snake_y + size_dir)
        window = np.zeros((size, size, 3), np.uint8)
        window[:, :, :] = orange
        window[world_center[1], world_center[0]] = white
        if top_corner[1] < 0:
            wy = -1 * top_corner[1]
        else:
            wy = 0
        y = 0
        for col in self.state:
            x = 0
            if top_corner[0] < 0:
                wx = -1 * top_corner[0]
            else:
                wx = 0
            for val in col:
                if top_corner[0] <= x <= btm_corner[0] and top_corner[1] <= y <= btm_corner[
                    1] and wy < size and wx < size:
                    window[wy, wx] = val
                    wx += 1
                if x > btm_corner[0]:
                    break
                x += 1
            if top_corner[1] <= y <= btm_corner[1]:
                wy += 1
            if y > btm_corner[1]:
                break
            y += 1

        return window

    def getHead(self):
        self.snake_x = self.env.head[0]
        self.snake_y = self.env.head[1]

    def cvt(self, tup):
        cvt_l = [self.grid_object.HEAD_COLOR, self.grid_object.BODY_COLOR, self.grid_object.FOOD_COLOR,
                 self.grid_object.SPACE_COLOR, orange]
        cvt_l = list(map(tuple, cvt_l))
        return cvt_l.index(tup)

    def getState(self):
        # self.state = cv2.resize(self.state, fx=0.1, fy=0.1, dsize=(0, 0))
        if not settings.WINDOW_VIEW_MODE:
            self.stateExp = self.state[:]
        else:
            self.stateExp = self.state[:]
            self.state = self.getWindow()
        if not USE_PIXELS:
            # DO STUFF WITH STATE FIX
            n_state = np.array(self.state)
            n_state.resize((n_state.shape[1] * n_state.shape[0], len(n_state)))
            n_state = list(map(tuple, n_state))
            n_state = list(map(self.cvt, n_state))
            self.ret_state = n_state
        else:
            self.ret_state = self.state
        self.ret_state = np.array(self.ret_state)

    def getGameDisplay(self):
        self.display = self.getDisplay()
        self.write_info()

    def render(self, sleep=True, fps=FPS):
        self.getGameDisplay()
        cv2.imshow(WIN_NAME, self.display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
        if sleep:
            et = time.time()
            sleeptime = (1 / fps) - (et - self.st)
            if sleeptime < 0:
                sleeptime = 0
                print(sleeptime)
            time.sleep(sleeptime)


if __name__ == '__main__':
    envf = SnakeEnv()
    envf.reset()
    envf.step(1)
    envf.render()
