import pygame
import numpy as np
from typing import Tuple, List
from random import random
from vector import *

LINE_SKIP = 10
MAX_DEPTH = 10

# ASPECT_RATIO = 1/1
ASPECT_RATIO = 16/9

WIDTH  = 400
HEIGHT = int(WIDTH / ASPECT_RATIO)

SAMPLES_PER_PIXEL = 25;

ORIGIN = np.array([0,0,0])




class Ray:
    def __init__(self, origin: np.ndarray = ORIGIN, direction: np.ndarray = np.array([0,0,0])) -> None:
        self.origin = origin
        self.direction = direction

    def at(self, t: float) -> np.ndarray:
        return self.origin + t*self.direction


class HitRecord:
    def __init__(self, p: np.ndarray = np.array([0,0,0]), normal: np.ndarray = np.array([1,1,1]), t: float = 0, hit: bool = False, frontFace:bool = True) -> None:
        self.p = p
        self.normal = normal
        self.t = t 
        self.hit = hit
        self.frontFace = frontFace


    def setFaceNormal(self, r: Ray, outwardNormal: np.ndarray) -> None:
        self.frontFace = r.direction.dot(outwardNormal) < 0
        
        if self.frontFace:
            self.normal = outwardNormal
        else:
            self.normal = outwardNormal * -1


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
        c = oc.dot(oc) - self.radius*self.radius
        discriminant = half_b*half_b - a*c

        if discriminant < 0:
            return HitRecord()

        sqrtd = np.sqrt(discriminant)

        root = (-half_b - sqrtd) / a

        if (root < t_min) or (t_max < root):
            root = (-half_b + sqrtd) / a

            if (root < t_min) or (t_max < root):
                return HitRecord()

        rec = HitRecord(r.at(root), (r.at(root) - self.center) / self.radius, root, True)

        

        outwardNormal = (r.at(root) - self.center) / self.radius

        rec.setFaceNormal(r, outwardNormal)


        return rec


class HittableList(Hittable):
    def __init__(self) -> None:
        super().__init__()
        self.objects: List[Hittable] = []

    def append(self, object: Hittable) -> None:
        self.objects.append(object)

    def clear(self) -> None:
        self.objects.clear()

    def hit(self, r: Ray, t_min: float, t_max: float) -> HitRecord:
        closestSoFar = t_max

        rec = HitRecord()
        for object in self.objects:
            tmp_rec = object.hit(r, t_min, closestSoFar)
            if tmp_rec.hit :
                closestSoFar = tmp_rec.t
                rec = tmp_rec

        return rec


class Camera:
    def __init__(self, aspectRatio: float) -> None:
        self.viewportHeight = 2
        self.viewportWidth = self.viewportHeight * aspectRatio
        self.focalLenght = 1

        self._origin = np.array([0,0,0])
        self._horizontal = np.array([self.viewportWidth, 0, 0])
        self._vertical = np.array([0, self.viewportHeight, 0])
        self._lowerLeftCorner = self._origin - self._horizontal/2 - self._vertical/2 - np.array([0, 0, self.focalLenght])

    def getRay(self, u: float, v: float) -> Ray:
        return Ray(self._origin, self._lowerLeftCorner + u*self._horizontal + v*self._vertical - self._origin)


def rayColor(ray: Ray, world: Hittable, depth: int) -> np.ndarray:
    if depth <= 0:
        return newVector(0,0,0)

    rec = world.hit(ray, 0, np.Infinity)
    if rec.hit :
        target = rec.p + rec.normal + randomVectorInUnitSphere()
        return 0.5 * rayColor(Ray(rec.p, target - rec.p), world, depth -1)

    unit_direction = ray.direction / np.linalg.norm(ray.direction)
    t = 0.5 * (unit_direction[1] + 1)
    color = (1 - t)*np.array([1,1,1]) + t*np.array([0.5, 0.7, 1])
    return color


def correctColor(color: np.ndarray) -> np.ndarray:
    r = color[0]
    g = color[1]
    b = color[2]

    scale = 1 / SAMPLES_PER_PIXEL

    r = np.clip(np.sqrt(r * scale),0,0.999)
    g = np.clip(np.sqrt(g * scale),0,0.999)
    b = np.clip(np.sqrt(b * scale),0,0.999)

    return np.array([r, g ,b])


def main():
    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    pygame.display.set_caption("Raytracing")

    cam = Camera(ASPECT_RATIO)

    world = HittableList()

    world.append(Sphere(np.array([0, 0, -1]), 0.5))
    world.append(Sphere(np.array([0, -100.5, -1]), 100))
    # world.append(Sphere(np.array([1, 0, -2]), 0.7))
    
    screen.fill((0,0,0))
    for y in range(HEIGHT):
        for x in range(WIDTH):

            pixelColor = np.array([0,0,0], dtype=float)
            for _ in range(SAMPLES_PER_PIXEL) :
                u = (x + random())/(WIDTH-1)
                v = (y + random())/(HEIGHT-1)

                r = cam.getRay(u, v)
                pixelColor += rayColor(r, world, MAX_DEPTH)

            pixelColor = correctColor(pixelColor)

            # TODO: Potentially flip the axis
            screen.set_at((WIDTH-x, HEIGHT-y), (pixelColor*255).tolist())

        if y % LINE_SKIP == 0 :
            pygame.display.flip()

    running = True
    while running :
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

if __name__ == "__main__":
    main()