import math
from random import uniform
from .base import VersionSpace


PI = math.pi
TWO_PI = 2 * math.pi

class Angle(object):
    def __init__(self, theta, unit='deg'):
        if unit == 'deg':
            self.deg = theta % 360.0
            self.rad = self.deg * PI / 180.0

        elif unit == 'rad':
            self.rad = theta % TWO_PI
            self.deg = self.rad * 180.0 / PI

        else:
            raise ValueError("Unsupported unit type '{0}'. Only 'rad' and 'deg' available.".format(unit))

    def __repr__(self):
        return "Rad: {rad}, Deg: {deg}".format(rad=self.rad, deg=self.deg)

    def to_point(self, radius=1.0):
        if radius <= 0:
            raise ValueError("Radius must be positive!")
        return radius * Point(math.cos(self.rad), math.sin(self.rad))


class Point(object):
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

    @classmethod
    def from_iterable(cls, obj):
        if len(obj) != 2:
            pass
            #raise ValueError("Object has length != 2.")
        return cls(obj[0], obj[1])

    @classmethod
    def from_angle(cls, theta):
        return cls(math.cos(theta), math.sin(theta))

    def __str__(self):
        return "({x:.3f}, {y:.3f})".format(x=self.x, y=self.y)

    def __repr__(self):
        return self.__str__()

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return self.__radd__(other)

        return Point(self.x + other.x, self.y + other.y)

    def __radd__(self, other):
        if not isinstance(other, (int, float)):
            raise ValueError("Operation of Point with {0} not supported.".format(type(other)))

        return Point(other + self.x, other + self.y)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return self.__radd__(-other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return self.__rmul__(other)

        return self.x * other.x + self.y * other.y

    def __rmul__(self, other):
        if not isinstance(other, (int, float)):
            raise ValueError("Operation of Point with {0} not supported.".format(type(other)))

        return Point(other * self.x, other * self.y)

    def __div__(self, other):
        if not isinstance(other, (int, float)):
            raise ValueError("Operation of Point with {0} not supported.".format(type(other)))

        if other == 0:
            raise ZeroDivisionError

        return Point(self.x / other, self.y / other)

    def __truediv__(self, other):
        return self.__div__(other)

    def __neg__(self):
        return Point(-self.x, -self.y)

    def rotate(self, theta):
        """ Counter-clockwise rotation of point """
        cos_th = math.cos(theta.rad)
        sin_th = math.sin(theta.rad)
        return Point(self.x * cos_th - self.y * sin_th, self.x * sin_th + self.y * cos_th)

    def orthogonal(self, clockwise=False):
        if clockwise:
            return Point(self.y, -self.x)
        return Point(-self.y, self.x)

    @property
    def angle(self):
        return Angle(theta=math.atan2(self.y, self.x), unit='rad')

    @property
    def norm(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def normalize(self):
        return self.__div__(self.norm)


class Circle(VersionSpace):
    def __init__(self):
        super().__init__(n_samples=0)
        self.__left_limit = Point(1, 0)
        self.__right_limit = Point(1, 0)
        self.__was_cut = False

    def __repr__(self):
        return "Limits: left = {left}, right = {right}, Size: {size:.5f}".format(left=self.left_limit,
                                                                                 right=self.right_limit,
                                                                                 size=self.volume)

    @property
    def left_limit(self):
        return self.__left_limit

    @property
    def right_limit(self):
        return self.__right_limit

    def clear(self):
        super().clear()
        self.__init__()

    @property
    def volume(self):
        if self.__was_cut:
            scalar_prod = self.__left_limit * self.__right_limit
            clipped_prod = max(-1, min(1, scalar_prod))
            return math.acos(clipped_prod) % TWO_PI
        return TWO_PI

    def __first_cut(self, point):
        self.__left_limit = point.orthogonal(clockwise=True)
        self.__right_limit = point.orthogonal(clockwise=False)

    def __posterior_cut(self, point):
        left_limit_correctly_classified = self.left_limit * point >= 0
        right_limit_correctly_classified = self.right_limit * point >= 0

        if not left_limit_correctly_classified and not right_limit_correctly_classified:
            raise RuntimeError("Point incompatible with current version space!")

        if not left_limit_correctly_classified and right_limit_correctly_classified:
            self.__left_limit = point.orthogonal(clockwise=True)

        if left_limit_correctly_classified and not right_limit_correctly_classified:
            self.__right_limit = point.orthogonal(clockwise=False)

    def cut(self, point):
        point = point.normalize()

        if not self.__was_cut:
            self.__first_cut(point)
            self.__was_cut = True
        else:
            self.__posterior_cut(point)

    def update(self, point, label):
            self.cut(Point.from_iterable(label * point))

    def angle_limits(self):
        theta_left = self.left_limit.angle.rad

        theta_right = self.right_limit.angle.rad
        if theta_right <= theta_left:
            theta_right += TWO_PI

        return theta_left, theta_right

    def sample(self, n_samples):
        theta_left, theta_right = self.angle_limits()
        thetas = [uniform(theta_left, theta_right) for _ in range(n_samples)]
        return [Point.from_angle(theta) for theta in thetas]

    def is_inside(self, points):
        theta_left, theta_right = self.angle_limits()
        if isinstance(points, Point):
            return theta_left <= points.angle.rad <= theta_right
        return [theta_left <= point.angle.rad <= theta_right for point in points]

