import glob
import sys
import time
from threading import Thread

import cv2
import natsort
import numpy as np

import SnakeEnv
import settings
from get_keys import KeyChecker

print_error = True
speed = 1
FPS = settings.FPS

keyChecker = KeyChecker()


def im_write(img, text, coords):
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_color = (255, 255, 255)
    line_size = 1
    line_type = cv2.LINE_AA
    cv2.putText(img, str(text),
                coords, font, line_size, line_color, lineType=line_type)
    return img


def write_info(info, display):
    for key in info:
        display = im_write(display, f"{key}: {info[key][0]}", info[key][1])
    return display


def getSpeed():
    global speed
    while True:
        if keyChecker.checkKey("1"):
            speed = 1
        elif keyChecker.checkKey("2"):
            speed = 5
        elif keyChecker.checkKey("3"):
            speed = 10
        elif keyChecker.checkKey("4"):
            speed = 100
        elif keyChecker.checkKey("5"):
            speed = 10000000000


speed_tracker = Thread(target=getSpeed, daemon=True)
speed_tracker.start()


def playVideo(file, files):
    global speed
    save = np.load(f"{file}", allow_pickle=True)
    for world_data in save[0]:
        st = time.time()
        world_info = world_data[0]
        new_frame = SnakeEnv.renderWorld(world_info, settings.pixels_per_unit, True, settings.RENDER_GAP)
        # new_frame = im_write(new_frame, save[0], (10, 715))
        info = world_data[1]
        new_frame = write_info(info, new_frame)
        new_frame = im_write(new_frame, save[1], (900, 1190))  # END REWARD
        # state = SnakeEnv.renderWorld(world_info, settings.state_pixels_per_unit, True, settings.STATE_GAP)
        cv2.imshow("hist", new_frame)
        # cv2.imshow("sensor", state)
        if cv2.waitKey(1) and keyChecker.checkKey("N"):
            break
        elif keyChecker.checkKey("K"):
            cv2.destroyWindow('hist')
            return True
        elif keyChecker.checkKey("B"):
            playVideo(files[-1], files)
        else:
            fps = FPS * speed
            et = time.time()
            sleeptime = (1 / fps) - (et - st)
            if sleeptime < 0:
                sleeptime = 0
            time.sleep(sleeptime)


def show_video():
    global speed
    global keyChecker
    keyChecker = KeyChecker()
    path = settings.GAME_SAVE
    files = [f for f in glob.glob(path + "**/*.npy", recursive=True)]
    files = natsort.natsorted(files)
    time.sleep(0.1)
    for file in files:
        try:
            shouldBreak = playVideo(file, files)
            if shouldBreak:
                break
        except Exception as e:
            if print_error:
                print(e)

    time.sleep(1)
    cv2.destroyWindow('hist')
    sys.exit()


if __name__ == '__main__':
    show_video()
