"""
MIT License

Copyright (c) 2023 Penn Jenks

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
from skimage.measure import find_contours
from shapely import Polygon, GeometryCollection


def as_polygon(gray: np.ndarray) -> Polygon:

    def to_righthanded(xy):
        xy = np.roll(xy, 1, -1)
        xy[:, 1] *= -1
        xy[:, 1] += 1
        return xy

    regions = find_contours(gray, 128)
    # coordinate values start between [-.5 and 28.5]
    # and we want to put them in range [0, 1]
    polygons = [Polygon(to_righthanded((region + .5) / 29)) for region in regions]

    result = Polygon()
    for poly in polygons:
        if result.intersects(poly):
            result = result.difference(poly)
        else:
            result = result.union(poly)
    return result


path = 'mnist.npz'
out = {'x_train': [], 'x_test': []}
with np.load(path) as data:
    for split in ('x_train', 'x_test'):
        examples = data[split]
        for example in examples:
            poly = as_polygon(example)
            out[split].append(poly)

        out[split] = GeometryCollection(out[split]).wkb
    out['y_train'] = data['y_train']
    out['y_test'] = data['y_test']

np.savez_compressed('polygon_mnist.npz', **out)

"""
# generate plot
import matplotlib.pyplot as plt
import shapely.wkb

with np.load('polygon_mnist.npz') as data:
    train_polygons = shapely.wkb.loads(data['x_train'])



def plot_polygon(ax, poly):
    ax.fill(*poly.exterior.xy, facecolor='gray', edgecolor='black')
    for hole in poly.interiors:
        ax.fill(*hole.xy, facecolor='white', edgecolor='black')

fig, axs = plt.subplots(4, 6, figsize=(10, 6))

for ax, poly in zip(axs.ravel(), train_polygons.geoms):
    if isinstance(poly, MultiPolygon):
        for poly in poly.geoms:
            plot_polygon(ax, poly)
    else:
        plot_polygon(ax, poly)
    ax.axis('off')
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

plt.show()
"""