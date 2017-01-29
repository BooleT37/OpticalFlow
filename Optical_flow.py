import numpy as np
import cv2
import random
import sys
import argumentsParser
import math

MIN_BUBBLE_RADIUS = 20
MAX_BUBBLE_RADIUS = 40
BUBBLE_DENSITY = 0.1


class Color:
    def __init__(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b

    def __str__(self):
        return "(R: {}, G: {}, B)".format(self.r, self.g, self.b)


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return "(x: {}, y: {})".format(self.x, self.y)


class Bubble:
    def __init__(self, color, center, radius, speed=(0, 0)):
        self.color = color
        self.center = center
        self.radius = radius
        self.speed = speed
        self.mass = BUBBLE_DENSITY * math.pi * (radius ** 2)

    def __str__(self):
        return "(center: {}, radius: {})".format(self.center, self.radius)


class ShiftedArray:
    def __init__(self, array, shift):
        self.array = array
        self.shift = shift

    def __getitem__(self, indexes):
        if indexes[0] + self.shift[0] >= self.array.shape[0] or indexes[1] + self.shift[1] >= self.array.shape[1]:
            raise Exception("Getting array element out of bounds!")
        return self.array[indexes[0] + self.shift[0], indexes[1] + self.shift[1]]


# noinspection SpellCheckingInspection,PyPep8Naming
class App:
    def run(self, options):
        cam = cv2.VideoCapture(options["file"])

        ret, prev = cam.read()
        if not ret:
            return

        def shiftBubble(bubble, flow):
            vector = self.getVectorForBubble(bubble, flow)

            acc = vector[0] / bubble.mass, vector[1] / bubble.mass

            newCenter = Point(
                int(bubble.center.x + bubble.speed[0] + acc[0] / 2),
                int(bubble.center.y + bubble.speed[1] + acc[1] / 2)
            )
            newSpeed = (bubble.speed[0] + acc[0], bubble.speed[1] + acc[1])
            if (newCenter.x < flow.shape[1] - bubble.radius)\
                    and (newCenter.y < flow.shape[0] - bubble.radius) \
                    and (newCenter.x > bubble.radius) \
                    and (newCenter.y > bubble.radius):
                bubble.center = newCenter
                bubble.speed = newSpeed
                return bubble
            else:
                return None

        bubbles = self.generateRandomBubbles(prev.shape)
        self.drawBubbles(prev, bubbles)
        prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

        ret = True
        while ret:
            ret, img = cam.read()
            if ret:
                # try to show initial img with filled circles
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(prevgray, gray, 0.5, 3, 15, 3, 5, 1.2, 0)
                bubbles = list(filter(lambda b: b is not None, (map(lambda b: shiftBubble(b, flow), bubbles))))
                self.drawBubbles(img, bubbles)
                prevgray = gray
                cv2.imshow('flow', self.drawFlow(gray, flow))
                ch = cv2.waitKey(5)
                if ch == 27:
                    break
        cv2.destroyAllWindows()

    @staticmethod
    def drawBubbles(img, bubbles):
        for bubble in bubbles:
            cv2.circle(img, (bubble.center.x, bubble.center.y), bubble.radius, (bubble.color.r, bubble.color.g, bubble.color.b), -1)
        cv2.imshow('bubbles', img)

    @staticmethod
    def generateRandomBubbles(radius):
        h, w = radius[:2]
        n = 20
        bubbles = []
        for i in range(0, n - 1):
            color = Color(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            radius = random.randint(MIN_BUBBLE_RADIUS, MAX_BUBBLE_RADIUS)
            center = Point(x=random.randint(radius, w - radius), y=random.randint(radius, h - radius))
            bubbles.append(Bubble(color, center, radius))
        return bubbles

    @staticmethod
    def drawFlow(img, flow, step=16):
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

    @staticmethod
    def getVectorForBubble(bubble, flow):
        step = 10
        flow = ShiftedArray(flow, (bubble.center.y, bubble.center.x))

        def addVector(vector1, vector2):
            return vector1[0] + vector2[0], vector1[1] + vector2[1]

        def processRow(x, y, radiusSquare, vector):
            while x < 0:
                vector = addVector(vector, flow[y, x])
                vector = addVector(vector, flow[y, -x])
                vector = addVector(vector, flow[-y, x])
                vector = addVector(vector, flow[-y, -x])
                x += step
                nextY = y + step
                if x ** 2 + nextY ** 2 <= radiusSquare:
                    vector = processRow(x, nextY, radiusSquare, vector)
            return int(vector[0]), int(vector[1])

        return processRow(-bubble.radius, 0, bubble.radius ** 2, (0, 0))


app = App()

app.run(argumentsParser.parsearguments(sys.argv[1:]))
