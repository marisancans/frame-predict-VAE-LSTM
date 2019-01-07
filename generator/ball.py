import cv2
import numpy as np
import random as random
import time
import os

height, width = 32, 32
img = np.zeros((height, width, 3), np.uint8)
img[:, :] = [255, 255, 255]


for d in range(1000):
    print(d)
    r = 1
    x, y, t = 0, 0, 0
    vx = 4  # Velocity along x-axis
    vy = 0  # Velocity along y-axis
    g = 1   # Acceleration due to gravity
    c = 0
    offset = random.randint(1, height/2)

    while True:
        t += 1
        img_out = img.copy()
        cv2.circle(img_out, (int(x), int(abs(y))+r+offset), r, (0, 0, 255), -1)
        x = vx * t
        y = vy * t - 1 / 2 * g * t ** 2

        if abs(y) > height - r - offset:
            break

        path = '/home/maris/rnn'
        folder = str(d) # format(int(time.time()))
        if not os.path.exists(path + '/' + folder):
            os.makedirs(path + '/' + folder)

        cv2.imwrite(path + '/' + folder + '/' + str(c) + '.png', img_out)
        c += 1
