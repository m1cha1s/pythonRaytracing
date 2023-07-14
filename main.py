import numpy as np
from random import random
from typing import List, Tuple
from vector import *
from multiprocessing import Pool
import pygame
import ray

ray.init()

MAX_DEPTH = 10

# ASPECT_RATIO = 1/1
ASPECT_RATIO = 16 / 9

WIDTH = 400
HEIGHT = int(WIDTH / ASPECT_RATIO)

SAMPLES_PER_PIXEL = 100

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

    def hit(self, r: Ray, t_min: float, t_max: float) -> HitRecord:
        return HitRecord()


class Sphere(Hittable):
    def __init__(self, center: np.ndarray, radius: float) -> None:
        super().__init__()

        self.center = center
        self.radius = radius

    def hit(self, r: Ray, t_min: float, t_max: float) -> HitRecord:
        oc = r.origin - self.center
        a = r.direction.dot(r.direction)
        half_b = oc.dot(r.direction)
        c = oc.dot(oc) - self.radius * self.radius
        discriminant = half_b * half_b - a * c

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


class HittableList(Hittable):
    def __init__(self) -> None:
        super().__init__()
        self.objects: List[Hittable] = []

    def append(self, obj: Hittable) -> None:
        self.objects.append(obj)

    def clear(self) -> None:
        self.objects.clear()

    def hit(self, r: Ray, t_min: float, t_max: float) -> HitRecord:
        closest_so_far = t_max

        rec = HitRecord()
        for obj in self.objects:
            tmp_rec = obj.hit(r, t_min, closest_so_far)
            if tmp_rec.hit:
                closest_so_far = tmp_rec.t
                rec = tmp_rec

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

    def get_ray(self, u: float, v: float) -> Ray:
        return Ray(self._origin, self._lowerLeftCorner + u * self._horizontal + v * self._vertical - self._origin)


def ray_color(ray: Ray, world: Hittable, depth: int) -> np.ndarray:
    if depth <= 0:
        return newVector(0, 0, 0)

    rec = world.hit(ray, 0, np.Infinity)
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
            yield (x, y)


def uv(x: int, y: int, width: int, height: int) -> Tuple[float, float]:
    u = (x + random()) / (width - 1)
    v = (y + random()) / (height - 1)
    return u, v


@ray.remote
def getPixelColor(x: int, y: int, cam: Camera, world: HittableList) -> Tuple[int, int, np.ndarray]:
    pixel_color = np.array([0, 0, 0], dtype=float)
    for _ in range(SAMPLES_PER_PIXEL):
        pixel_uv = uv(x, y, WIDTH, HEIGHT)
        r = cam.get_ray(*pixel_uv)

        pixel_color += ray_color(r, world, MAX_DEPTH)
    return x, y, correct_color(pixel_color)


def main():
    # Pygame stuff
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Raytracing")

    cam = Camera(ASPECT_RATIO)

    world = HittableList()

    world.append(Sphere(np.array([0, 0, -1]), 0.5))
    world.append(Sphere(np.array([0, -100.5, -1]), 100))
    # world.append(Sphere(np.array([1, 0, -2]), 0.7))

    screen.fill((0, 0, 0))
    pygame.display.flip()

    future_pixels = [getPixelColor.remote(p[0], p[1], cam, world) for p in xy(WIDTH, HEIGHT)]

    pixel_colors = ray.get(future_pixels)

    for pixel in pixel_colors:
        pixel_color = pixel[2]
        screen.set_at((WIDTH - pixel[0], HEIGHT - pixel[1]), (pixel_color * 255).tolist())

    pygame.display.flip()

    running = True
    while running:
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False


if __name__ == "__main__":
    main()
