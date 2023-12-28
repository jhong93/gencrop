ID_COUNTER = 0

class Box:

    def __init__(self, x, y, w, h, score=-1, payload=None):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self._score = score
        self._payload = payload

        global ID_COUNTER
        ID_COUNTER += 1

        self._id = ID_COUNTER

    @property
    def area(self):
        return self.w * self.h

    @property
    def score(self):
        return self._score

    @property
    def payload(self):
        return self._payload

    @property
    def x2(self):
        return self.x + self.w

    @property
    def y2(self):
        return self.y + self.h

    @property
    def cx(self):
        return self.x + self.w / 2

    @property
    def cy(self):
        return self.y + self.h / 2

    def iou(self, b2):
        ia = self.intersect(b2).area
        assert ia >= 0
        if ia > 0:
            ia /= self.area + b2.area - ia
        return ia

    def intersect(self, b2):
        ix1, iy1 = max(self.x, b2.x), max(self.y, b2.y)
        ix2, iy2 = min(self.x + self.w, b2.x + b2.w), \
            min(self.y + self.h, b2.y + b2.h)
        iw, ih = max(ix2 - ix1, 0), max(iy2 - iy1, 0)
        return Box(ix1, iy1, iw, ih)

    def union(self, b2):
        x1 = min(self.x, b2.x)
        y1 = min(self.y, b2.y)
        x2 = max(self.x + self.w, b2.x + b2.w)
        y2 = max(self.y + self.h, b2.y + b2.h)
        return Box(x1, y1, x2 - x1, y2 - y1)

    def contains(self, b2):
        return (self.contains_point(b2.x, b2.y)
                and self.contains_point(b2.x + b2.w, b2.y)
                and self.contains_point(b2.x, b2.y + b2.h)
                and self.contains_point(b2.x + b2.w, b2.y + b2.h))

    def contains_point(self, x, y):
        return (x >= self.x and x <= self.x + self.w
                and y >= self.y and y <= self.y + self.h)

    def __eq__(self, b2):
        return (self.x == b2.x and self.y == b2.y
                and self.w == b2.w and self.h == b2.h)

    @property
    def xywh(self):
        return self.x, self.y, self.w, self.h

    @property
    def xyxy(self):
        return self.x, self.y, self.x2, self.y2

    def __repr__(self):
        return str(self.xywh)


def xyxy2xywh(a):
    return a[0], a[1], a[2] - a[0], a[3] - a[1]
