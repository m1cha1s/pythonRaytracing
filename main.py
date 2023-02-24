import numpy as np
from random import random
from typing import List, Tuple
from vector import *
import pygame

LINE_SKIP = 10
MAX_DEPTH = 3

# ASPECT_RATIO = 1/1
ASPECT_RATIO = 16 / 9

WIDTH = 400
HEIGHT = int(WIDTH / ASPECT_RATIO)

SAMPLES_PER_PIXEL = 10

ORIGIN = np.array([0, 0, 0])


class Ray:
    def __init__(self, origin: np.ndarray = ORIGIN, direction: np.ndarray = np.array([0, 0, 0])) -> None:
        self.origin = origin
        self.direction = direction

    def at(self, t: float) -> np.ndarray:
        return self.origin + t * self.direction


class HitRecord:
    def __init__(self, p: np.ndarray = np.array([0, 0, 0]), normal: np.ndarray = np.array([1, 1, 1]), t: float = 0,
                 hit: bool = False, front_face: bool = True) -> None:
        self.p = p
        self.normal = normal
        self.t = t
        self.hit = hit
        self.frontFace = front_face

    def set_face_normal(self, r: Ray, outward_normal: np.ndarray) -> None:
        self.frontFace = r.direction.dot(outward_normal) < 0

        if self.frontFace:
            self.normal = outward_normal
        else:
            self.normal = outward_normal * -1


class Hittable:
    def __init__(self) -> None:
        pass

    def hit(self, r: Ray, t_min: float, t_max: float) -> List[HitRecord]:
        raise NotImplemented


class Sphere(Hittable):
    def __init__(self, center: np.ndarray, radius: float) -> None:
        super().__init__()

        self.center = center
        self.radius = radius

    def _process_discrimninant(self, r, a, half_b, t_min: float, t_max: float, discriminant: float) -> HitRecord:
        if discriminant < 0:
            return HitRecord()

        sqrtd = np.sqrt(discriminant)

        root = (-half_b - sqrtd) / a

        if (root < t_min) or (t_max < root):
            root = (-half_b + sqrtd) / a

            if (root < t_min) or (t_max < root):
                return HitRecord()

        rec = HitRecord(r.at(root), (r.at(root) - self.center) / self.radius, root, True)

        outward_normal = (r.at(root) - self.center) / self.radius

        rec.set_face_normal(r, outward_normal)

        return rec

    def hit(self, r: Ray, t_min: float, t_max: float) -> List[HitRecord]:
        oc = r.origin - self.center
        # a = r.direction.dot(r.direction)
        a = np.apply_along_axis(lambda x: x.dot(x), axis=1, arr=r.direction)
        half_b = np.apply_along_axis(lambda x: oc.dot(x), axis=1, arr=r.direction)
        c = oc.dot(oc) - self.radius * self.radius
        discriminant = half_b * half_b - a * c

        vpd = np.vectorize(self._process_discrimninant)

        return vpd(r, a, half_b, t_min, t_max, discriminant)


class HittableList(Hittable):
    def __init__(self) -> None:
        super().__init__()
        self.objects: List[Hittable] = []

    def append(self, obj: Hittable) -> None:
        self.objects.append(obj)

    def clear(self) -> None:
        self.objects.clear()

    def hit(self, r: Ray, t_min: float, t_max: float) -> List[HitRecord]:
        closest_so_far: List[float] = [t_max for _ in r.direction]

        rec: List[HitRecord] = [HitRecord() for _ in r.direction]
        for obj in self.objects:
            tmp_rec = obj.hit(r, t_min, closest_so_far)
            for idx, tmp_rec_hit in enumerate(tmp_rec):
                if tmp_rec_hit.hit:
                    closest_so_far[idx] = tmp_rec_hit.t
                    rec[idx] = tmp_rec_hit

        return rec



class Camera:
    def __init__(self, aspect_ratio: float) -> None:
        self.viewportHeight = 2
        self.viewportWidth = self.viewportHeight * aspect_ratio
        self.focalLength = 1

        self._origin = np.array([0, 0, 0])
        self._horizontal = np.array([self.viewportWidth, 0, 0])
        self._vertical = np.array([0, self.viewportHeight, 0])
        self._lowerLeftCorner = self._origin - self._horizontal / 2 - self._vertical / 2 - np.array(
            [0, 0, self.focalLength])

    def get_ray(self, u: np.ndarray, v: np.ndarray) -> Ray:
        return Ray(self._origin, self._lowerLeftCorner + u * self._horizontal + v * self._vertical - self._origin)


def ray_color(ray: Ray, world: Hittable, depth: int) -> np.ndarray:
    if depth <= 0:
        return newVector(0, 0, 0)

    recs = world.hit(ray, 0, np.Infinity)
    for rec in recs:
        if rec.hit:
            target = rec.p + rec.normal + randomVectorInUnitSphere()
            return ray_color(Ray(rec.p, target - rec.p), world, depth - 1) * 0.5

    unit_direction = ray.direction / np.linalg.norm(ray.direction)
    t = 0.5 * (unit_direction[1] + 1)
    color = (1 - t) * np.array([1, 1, 1], dtype=float) + (t * np.array([0.5, 0.7, 1.0], dtype=float))
    return color


def correct_color(color: np.ndarray) -> np.ndarray:
    scale = 1 / SAMPLES_PER_PIXEL
    color = np.clip(np.sqrt(color * scale), 0, 0.999)
    return color


def xy(width: int, height: int) -> Tuple[int, int]:
    for y in range(height):
        for x in range(width):
            yield x, y


def uv(x: int, y: int, width: int, height: int) -> Tuple[float, float]:
    u = (x + random()) / (width - 1)
    v = (y + random()) / (height - 1)
    return u, v


def main():
    # Pygame stuff
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Raytracing")

    cam = Camera(ASPECT_RATIO)

    world = HittableList()

    world.append(Sphere(np.array([0, 0, -1]), 0.5))
    world.append(Sphere(np.array([0, -100.5, -1]), 100))
    world.append(Sphere(np.array([1, 0, -2]), 0.7))

    screen.fill((0, 0, 0))
    pygame.display.flip()

    for i in range(SAMPLES_PER_PIXEL):
        # pixel_color = np.array([0, 0, 0], dtype=float)
        us = []
        vs = []
        for pixel in xy(WIDTH, HEIGHT):
            u, v = uv(*pixel, WIDTH, HEIGHT)
            us.append([u])
            vs.append([v])

        r = cam.get_ray(np.array(us), np.array(vs))

        pixel_color = ray_color(r, world, MAX_DEPTH)
        pixel_color = correct_color(pixel_color)

        screen.set_at((WIDTH - pixel[0], HEIGHT - pixel[1]), (pixel_color * 255).tolist())

        if pixel[1] % LINE_SKIP == 0:
            pygame.display.flip()

    running = True
    while running:
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False


if __name__ == "__main__":
    main()
