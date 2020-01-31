# Center point를 기준으로 viewport build
import numpy as np


def get_frame(view):
    return (int(view[0]), int(view[2])), (int(view[1]), int(view[3]))


def merge(left, right, frame):
    l = frame[left[2]:left[3], left[0]:left[1]]
    r = frame[right[2]:right[3], right[0]:right[1]]
    return np.append(l, r, 1)


class Viewport:
    def __init__(self, width, height):
        self.fov = 110 / 360  # (4:3, 90), (16:9, 108) HTC vive = 110
        self.center = np.array([width / 2, height / 2], dtype=np.int16)  # x,y
        # (x0, x1, y0, y1)
        self.VIDEO_WIDTH = int(width)
        self.VIDEO_HEIGHT = int(height)
        self.width = int(self.VIDEO_WIDTH * self.fov / 2)
        self.height = int(self.VIDEO_HEIGHT * self.fov / 2)
        self.view_dict = {"left": None, "right": None}
        self.view = None
        self.update()
        self._build_view()
        self.point = []

    def find_tile(self):
        n, m = 4, 8
        h = self.VIDEO_HEIGHT / n
        w = self.VIDEO_WIDTH / m
        points = self.get_rectangle_point()
        tiles = [0 for _ in range(n * m)]
        for point in points:
            p = (int(point[0] / w), int(point[1] / h), int(point[2] / w), int(point[3] / h))
            x = [_ * m for _ in range(p[1], p[3] + 1)]
            y = [_ for _ in range(p[0], p[2] + 1)]
            for i in y:
                for j in x:
                    tiles[j + i] = 1

            for i in range(p[0], p[2] + 1):
                tiles[x + i] = 1
                # TODO array out of index error
                tiles[y + i] = 1
        return tiles

    def _build_view(self):
        self.view = [self.center[0] - self.width, self.center[0] + self.width,
                     self.center[1] - self.height, self.center[1] + self.height]
        # set Y
        self.view[2] = self.view[2] if self.view[2] >= 0 else 0
        self.view[3] = self.view[3] if self.view[3] <= self.VIDEO_HEIGHT else self.VIDEO_HEIGHT

        # set X
        if self.view[0] < 0:
            left = [self.VIDEO_WIDTH + self.view[0], self.VIDEO_WIDTH] + self.view[2:]
            right = [0, self.view[1]] + self.view[2:]
        elif self.view[1] > self.VIDEO_WIDTH:
            left = [self.view[0], self.VIDEO_WIDTH] + self.view[2:]
            right = [0, self.view[1] - self.VIDEO_WIDTH] + self.view[2:]
        else:
            self.view = [int(i) for i in self.view]
            return
        self.view_dict['left'], self.view_dict['right'] = np.array(left, dtype=np.int), np.array(right, dtype=np.int)
        self.view = None

    def get_view(self, frame):
        if self.view is None:
            return merge(self.view_dict["left"], self.view_dict["right"], frame)
        else:
            return frame[self.view[2]:self.view[3], self.view[0]:self.view[1]]

    def get_rectangle_point(self):
        # form --> (x1, y1), (x2, y2)
        if self.view is not None:
            return [(self.view[0], self.view[2], self.view[1], self.view[3])]
        else:
            l = self.view_dict["left"]
            r = self.view_dict["right"]
            l_rec = (l[0], l[2], l[1], l[3])
            r_rec = (r[0], r[2], r[1], r[3])
            return l_rec, r_rec

    # move --> [x,x] 2x1 vector
    def move(self, v):
        v = np.reshape(v, [2])
        v[0], v[1] = v[0] * self.VIDEO_WIDTH, v[1] * self.VIDEO_HEIGHT
        self.center[0] += v[0]
        self.center[1] += v[1]
        self.update()
        self._build_view()

    def update(self):
        if self.center[0] < 0:
            self.center[0] += self.VIDEO_WIDTH
        elif self.center[0] > self.VIDEO_WIDTH:
            self.center[0] -= self.center[0]

        if self.center[1] < 0:
            self.center[1] = 0
        elif self.center[1] > self.VIDEO_HEIGHT:
            self.center[1] = self.VIDEO_HEIGHT
        self.center = [int(i) for i in self.center]

    def set_center(self, c, normalize=False):
        if normalize:
            c = c[0] * self.VIDEO_WIDTH, c[1] * self.VIDEO_HEIGHT
        self.center = np.array(c)
        self._build_view()
