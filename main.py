import pygame
import numpy as np

ASPECT_RATIO = 1/1

WIDTH  = 200
HEIGHT = int(WIDTH * ASPECT_RATIO)

VIEWPORT_HEIGHT = 2
VIEWPORT_WIDTH = VIEWPORT_HEIGHT * ASPECT_RATIO
FOCAL_LENGHT = 1

ORIGIN = np.array([0,0,0])
HORIZONTAL = np.array([VIEWPORT_WIDTH, 0, 0])
VERTICAL = np.array([0, VIEWPORT_HEIGHT, 0])
LOWER_LEFT_CORNER = ORIGIN - HORIZONTAL/2 - VERTICAL/2 - np.array([0, 0, FOCAL_LENGHT])


class Ray:
    def __init__(self, origin: np.ndarray = ORIGIN, direction: np.ndarray = np.array([0,0,0])) -> None:
        self.origin = origin
        self.direction = direction

    def at(self, t: float) -> np.ndarray:
        return self.origin + t*self.direction

class HitRecord:
    def __init__(self, p: np.ndarray, normal: np.ndarray, t: float) -> None:
        self.p = p
        self.normal = normal
        self.t = t 

class Hittable:
    def hit(self, r: Ray, t_min: float, t_max: float) -> HitRecord:
        pass

def hit_sphere(center: np.ndarray, radius: float, r: Ray) -> float:
    oc = r.origin - center
    a = np.dot(r.direction, r.direction)
    b = 2 * np.dot(oc, r.direction)
    c = np.dot(oc, oc) - radius*radius
    discriminant = b*b - 4*a*c
    if discriminant < 0:
        return -1
    else:
        return (-b - np.sqrt(discriminant)) / (2*a)


def ray_color(ray: Ray):
    t = hit_sphere(np.array([0,0,-1]), 0.5, ray)

    if t > 0:
        N = ray.direction / np.linalg.norm(ray.direction)
        return 0.5*np.array([N[0]+1, N[1]+1, N[2]+1])

    unit_direction = ray.direction / np.linalg.norm(ray.direction)
    t = 0.5 * (unit_direction[1] + 1)
    return (1 - t)*np.array([1,1,1]) + t*np.array([0.5, 0.7, 1])


def main():
    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    pygame.display.set_caption("Raytracing")

    
    screen.fill((0,0,0))
    for y in range(HEIGHT):
        for x in range(WIDTH):

            # Do the drawing here
            u = x/(WIDTH-1)
            v = y/(HEIGHT-1)

            r = Ray(ORIGIN, LOWER_LEFT_CORNER + u*HORIZONTAL + v*VERTICAL - ORIGIN)
            pixel_color = ray_color(r)
            # print(pixel_color)

            # TODO: Potentially flip the axis
            screen.set_at((WIDTH-x, HEIGHT-y), pixel_color*255)


            pygame.display.flip()

    running = True
    while running :
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

if __name__ == "__main__":
    main()