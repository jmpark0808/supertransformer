from PIL import Image
import os
import numpy as np
import math, random
from typing import List, Tuple
import matplotlib.pyplot as plt

def generate_polygon(center: Tuple[float, float], avg_radius: float,
                     irregularity: float, spikiness: float,
                     num_vertices: int) -> List[Tuple[float, float]]:
    """
    Start with the center of the polygon at center, then creates the
    polygon by sampling points on a circle around the center.
    Random noise is added by varying the angular spacing between
    sequential points, and by varying the radial distance of each
    point from the centre.

    Args:
        center (Tuple[float, float]):
            a pair representing the center of the circumference used
            to generate the polygon.
        avg_radius (float):
            the average radius (distance of each generated vertex to
            the center of the circumference) used to generate points
            with a normal distribution.
        irregularity (float):
            variance of the spacing of the angles between consecutive
            vertices.
        spikiness (float):
            variance of the distance of each vertex to the center of
            the circumference.
        num_vertices (int):
            the number of vertices of the polygon.
    Returns:
        List[Tuple[float, float]]: list of vertices, in CCW order.
    """
    # Parameter check
    if irregularity < 0 or irregularity > 1:
        raise ValueError("Irregularity must be between 0 and 1.")
    if spikiness < 0 or spikiness > 1:
        raise ValueError("Spikiness must be between 0 and 1.")

    irregularity *= 2 * math.pi / num_vertices
    spikiness *= avg_radius
    angle_steps = random_angle_steps(num_vertices, irregularity)

    # now generate the points
    points = []
    angle = random.uniform(0, 2 * math.pi)
    for i in range(num_vertices):
        radius = clip(random.gauss(avg_radius, spikiness), 0, 2 * avg_radius)
        point = (center[0] + radius * math.cos(angle),
                 center[1] + radius * math.sin(angle))
        points.append(point)
        angle += angle_steps[i]

    return points

def random_angle_steps(steps: int, irregularity: float) -> List[float]:
    """Generates the division of a circumference in random angles.

    Args:
        steps (int):
            the number of angles to generate.
        irregularity (float):
            variance of the spacing of the angles between consecutive vertices.
    Returns:
        List[float]: the list of the random angles.
    """
    # generate n angle steps
    angles = []
    lower = (2 * math.pi / steps) - irregularity
    upper = (2 * math.pi / steps) + irregularity
    cumsum = 0
    for i in range(steps):
        angle = random.uniform(lower, upper)
        angles.append(angle)
        cumsum += angle

    # normalize the steps so that point 0 and point n+1 are the same
    cumsum /= (2 * math.pi)
    for i in range(steps):
        angles[i] /= cumsum
    return angles

def clip(value, lower, upper):
    """
    Given an interval, values outside the interval are clipped to the interval
    edges.
    """
    return min(upper, max(value, lower))

from matplotlib.path import Path

def get_points(tupVerts):
    x, y = np.meshgrid(np.arange(256), np.arange(256)) # make a canvas with coordinates
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x,y)).T 

    p = Path(tupVerts) # make a polygon
    grid = p.contains_points(points)
    mask = grid.reshape(256,256)
    return mask





num_images = 5000
image_save_directory = '/mnt/hdd/Datasets/ToyDatasetV2/TR'
for i in range(num_images):
    image = np.random.randint(0, 255, [256, 256, 3])
    random_center_x = np.random.randint(50, 200)
    random_center_y = np.random.randint(50, 200)
    random_radius = np.random.randint(50, 100)
    random_vertices = np.random.randint(3, 10)
    random_colour = np.random.randint(0, 255, [3])
    vertices = generate_polygon(center=(random_center_x, random_center_y),
                            avg_radius=random_radius,
                            irregularity=0.35,
                            spikiness=0.2,
                            num_vertices=random_vertices)
    mask = get_points(vertices)
    idx = np.where(mask)
    image[idx[0], idx[1], :] = random_colour
    shape_noise = np.random.normal(0, 1, [len(idx[0]), 3])
    image = image.astype(np.float64)
    image[idx[0], idx[1], :] *= shape_noise
    image = np.clip(image, 0, 255)

    im = Image.fromarray((image).astype(np.uint8))
    im.save(os.path.join(image_save_directory, f"Image/{i}.jpg"))

    im = Image.fromarray((mask * 255).astype(np.uint8))
    im.save(os.path.join(image_save_directory, f"Mask/{i}.jpg"))


num_images = 5000
image_save_directory = '/mnt/hdd/Datasets/ToyDatasetV2/TE'
for i in range(num_images):
    image = np.random.randint(0, 255, [256, 256, 3])
    random_center_x = np.random.randint(50, 200)
    random_center_y = np.random.randint(50, 200)
    random_radius = np.random.randint(50, 100)
    random_vertices = np.random.randint(3, 10)
    random_colour = np.random.randint(0, 255, [3])
    vertices = generate_polygon(center=(random_center_x, random_center_y),
                            avg_radius=random_radius,
                            irregularity=0.35,
                            spikiness=0.2,
                            num_vertices=random_vertices)
    mask = get_points(vertices)
    idx = np.where(mask)
    image[idx[0], idx[1], :] = random_colour
    shape_noise = np.random.normal(0, 1, [len(idx[0]), 3])
    image = image.astype(np.float64)
    image[idx[0], idx[1], :] *= shape_noise
    image = np.clip(image, 0, 255)
    # 

    im = Image.fromarray((image).astype(np.uint8))
    im.save(os.path.join(image_save_directory, f"Image/{i}.jpg"))

    im = Image.fromarray((mask*255).astype(np.uint8))
    im.save(os.path.join(image_save_directory, f"Mask/{i}.jpg"))