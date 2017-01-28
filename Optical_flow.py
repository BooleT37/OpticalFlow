import numpy as np
import cv2
import random
import sys
import argumentsParser


class Color:
    def __init__(self, r=None, g=None, b=None):
        self.r = r
        self.g = g
        self.b = b


class Point:
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y


class Bubble:
    def __init__(self, color, point, size):
        self.color = color
        self.point = point
        self.size = size


class App:
    def run(self, options):
        cam = cv2.VideoCapture("./Clips/Test.avi")

        ret, prev = cam.read()
        if not ret:
            return

        bubbles = self.generate_random_bubbles(prev.shape)
        self.draw_bubbles(prev, bubbles)
        prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

        ret = True
        while ret:
            ret, img = cam.read()
            if ret:
                # try to show initial img with filled circles
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(prevgray, gray, 0.5, 3, 15, 3, 5, 1.2, 0)
                ###
                self.draw_bubbles(prev, bubbles)
                ###
                prevgray = gray
                cv2.imshow('flow', self.draw_flow(gray, flow))
                ch = cv2.waitKey(5)
                if ch == 27:
                    break

    @staticmethod
    def draw_bubbles(img, bubbles):
        for bubble in bubbles:
            cv2.circle(img, (bubble.point.x, bubble.point.y), bubble.size, (bubble.color.r, bubble.color.g, bubble.color.b), -1)
        cv2.imshow('bubbles', img)

    @staticmethod
    def generate_random_bubbles(size):
        h, w = size[:2]
        n = 20
        bubbles = []
        for i in range(0, n - 1):
            color = Color(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            size = random.randint(10, 20)
            point = Point(x=random.randint(size, w - size), y=random.randint(size, h - size))
            bubbles.append(Bubble(color, point, size))
        return bubbles

    @staticmethod
    def draw_flow(img, flow, step=16):
        h, w = img.shape[:2]
        y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.polylines(vis, lines, 0, (0, 255, 0))
        for (x1, y1), (x2, y2) in lines:
            cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
        return vis


    # filePath = options["filePath"] if options["fromFile"] else 0


cv2.destroyAllWindows()

app = App()
app.run(argumentsParser.parsearguments(sys.argv[1:]))
