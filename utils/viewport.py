# Center point를 기준으로 viewport build
import numpy as np


def get_frame(view):
    return (int(view[0]), int(view[2])), (int(view[1]), int(view[3]))


def merge(left, right, frame):
    l = frame[left[2]:left[3], left[0]:left[1]]
    r = frame[right[2]:right[3], right[0]:right[1]]
    return np.append(l, r, 1)


w, h = 3840, 1920


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
            # start, end
            return [(self.view[0], self.view[2]), (self.view[1], self.view[3])]
        else:
            l = self.view_dict["left"]
            r = self.view_dict["right"]
            return [(l[0], l[2]), (l[1], l[3]), (r[0], r[2]), (r[1], r[3])]  # l_start, l_end, r_start, r_end

    # move --> [x,x] 2x1 vector
    def move(self, v):
        if not np.shape(v) == (2,):
            v = np.reshape(v, [2])
        self.center[0] += v[0] * self.VIDEO_WIDTH
        self.center[1] += v[1] * self.VIDEO_HEIGHT
        self.update()
        self._build_view()

    def update(self):
        if self.center[0] < 0:
            self.center[0] += abs(self.center[0] // self.VIDEO_WIDTH) * self.VIDEO_WIDTH
        elif self.center[0] > self.VIDEO_WIDTH:
            self.center[0] -= (self.center[0] // self.VIDEO_WIDTH) * self.VIDEO_WIDTH

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


class TileViewport(Viewport):
    def __init__(self, width, height, n, m):
        super(TileViewport, self).__init__(width, height)
        self.n = n
        self.m = m
        self.tile_width = w / n
        self.tile_height = h / m
        self.tile = None

    def set_center(self, c, normalize=False):
        super(TileViewport, self).set_center(c, normalize=normalize)
        self.tile_update()

    def move(self, v):
        super(TileViewport, self).move(v)
        self.tile_update()

    def tile_update(self):
        # form --> (x1, y1), (x2, y2)
        self.tile = np.zeros([self.n, self.m])
        rec_point = self.get_rectangle_point()

        if self.view is None:  # rec point --> 2
            # 이 때 작동안함
            # print("view dict")
            # print(rec_point)
            x1, x2, y1, y2 = rec_point[0][0], rec_point[1][0], rec_point[0][1], rec_point[1][1]
            x11, x22, y11, y22 = rec_point[2][0], rec_point[3][0], rec_point[2][1], rec_point[3][1]

            _x1, _x2 = int(x1 // self.tile_width), int(x2 // self.tile_width)
            _y1, _y2 = int(y1 // self.tile_height), int(y2 // self.tile_height)
            self.tile[_x1:_x2 + 1, _y1:_y2 + 1] = 1

            _x11, _x22 = int(x11 // self.tile_width), int(x22 // self.tile_width)
            _y11, _y22 = int(y11 // self.tile_height), int(y22 // self.tile_height)
            self.tile[_x11:_x22 + 1, _y11:_y22 + 1] = 1
        else:
            # print("view")
            # print(rec_point)
            x1, x2, y1, y2 = rec_point[0][0], rec_point[1][0], rec_point[0][1], rec_point[1][1]
            # print(f"(x1, x2), (y1, y2) : ({x1}, {x2}), ({y1}, {y2}) {rec_point}")
            _x1, _x2 = int(x1 // self.tile_width), int(x2 // self.tile_width)
            _y1, _y2 = int(y1 // self.tile_height), int(y2 // self.tile_height)
            # print(f"(_x1, _x2), (_y1, _y2) : ({_x1}, {_x2}), ({_y1}, {_y2}) {rec_point}")

            self.tile[_x1:_x2 + 1, _y1:_y2 + 1] = 1
        # print(self.tile.transpose())

    def tile_info(self):
        point = []
        print(self.tile.transpose)
        for i in range(self.n):
            for j in range(self.m):
                if self.tile[i, j] == 1:
                    point.append(self.tile_point(i, j))
        return point

    def tile_point(self, i, j):
        x1, x2 = i * self.tile_width, (i + 1) * self.tile_width
        y1, y2, = j * self.tile_height, (j + 1) * self.tile_height
        return (int(x1), int(y1)), (int(x2), int(y2))
