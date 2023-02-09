import numpy as np
from random import random

def newVector(x: float, y: float, z: float) -> np.ndarray:
    return np.array([x, y, z])

def randomVector(min: float, max: float) -> np.ndarray:
    return newVector(min + random()*(max-min), min + random()*(max-min), min + random()*(max-min))

def randomVectorInUnitSphere() -> np.ndarray:
    while True:
        p = randomVector(-1, 1)
        if p.dot(p) >= 1:
            continue
        return p


def main():
    print(randomVector(-1, 1))

if __name__ == "__main__":
    main()