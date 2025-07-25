import numpy as np
from numba import jit, boolean
from typing import Union


def angle2D(vector, norm=None, k=np):
    """ Evaluate angle of (list of) 2D-vector(s) with respect to axis 0.

    :param vector: (List of) vector for which angle(s) is(are) to be evaluated.
    :param norm: (Optional) pre-evaluated element wise norm for each vector.
    :param k: Math backend for `sqrt` and `arccos` methods, defaults to `np` (tested for `torch`).
    :return: Element wise angle of each vector with axis 0.
    """
    if k is np:
        vector = np.asarray(vector)

    if norm is None:
        norm = k.sqrt((vector**2).sum(-1))

    return k.sign(vector[..., 1]) * k.arccos(vector[..., 0]/norm)


@jit(nopython=True)
def voronoi_neighbors(ridge_points: Union[list, np.ndarray], num_cells: int) -> np.ndarray:
    """ Get connectivity map of Voronoi `ridged_points` over a total number of `num_cells` cells.

    It is assumed, that the `ridge_points` where evaluated with `scipy.spatial.Voronoi` on a set of
+    coordinates/positions with a total size of `num_cells`.

    :return: Boolean array of shape `(num_cells, num_cells)`, each indey pair `i`, `j` specifying whether the
             respective cells are Neighbors with respect to `ridge_points` of a Voronoi tesselation.
     """
    cells = np.zeros((num_cells, num_cells), boolean)
    for pair in ridge_points:
        cells[pair[0], pair[1]] = True
        cells[pair[1], pair[0]] = True

    return cells


def transform_to_labframe(x: Union[tuple, list, np.ndarray], unit_cell: Union[tuple, list, np.ndarray]) -> np.ndarray:
    """ Transform `coordinates` on a grid specified via a `unit_cell` into lab-frame coordinates/positions
        according to `coordinates . unit_cell` in dot-product notation.

    :param x: List or 2D array of coordinate tuples, `[[x11, x12, ...], [x21, x22, ...], ...]`.
    :param unit_cell: List or 2D array of unit-cell vectors that represents a basis of the coordinates.
    :return: The unit-cell transform of the coordinates into lab-frame coordinates/positions.
    """
    return np.squeeze(np.atleast_2d(x).dot(unit_cell))


def transform_to_coords(x: Union[tuple, list, np.ndarray], unit_cell: Union[list, tuple, np.ndarray]) -> np.ndarray:
    """ Transform lab-frame `positions` to `coordinates` on a grid specified via a `unit_cell`
        (inverse to `transform_to_labframe`).

    :param x: List or 2D array of lab-frame coordinate tuples, `[[x11, x12, ...], [x21, x22, ...], ...]`.
    :param unit_cell: List or 2D array of unit-cell vectors that represents a basis of the `coordinates` grid.
    :return: Inverse unit-cell transform of the lab-frame `positions` into grid-based coordinates.
    """
    return np.squeeze(np.atleast_2d(x).dot(np.linalg.inv(unit_cell)))


def rotate(x: Union[tuple, list, np.ndarray],
           origin: Union[list, tuple, np.ndarray] = (0, 0),
           angle: float = 0.,
           to_radians: bool = False) -> np.ndarray:
    """ Rotate a set of `coordinates` around a given `origin` by a specific `angle` (either in radians or degree).

    :param x: List or 2D array of lab-frame coordinate tuples, `[[x11, x12, ...], [x21, x22, ...], ...]`.
    :param origin: Point of origin around which the rotation of `coordinates` is performed, defaults to `(0, 0)`.
    :param angle: Rotation-angle, either in radians (if `to_radians == False`) or in degrees (if `to_radians == True`),
                  defaults to `0`.
    :param to_radians: Boolean specifying whether the `angle` is given in radians (`to_radians == False`) or in
                       degrees (`to_radians == True`), defaults to `False`.
    :return: A set of coordinates, `x` rotated around the `origin` by the specified `angle`.
    """
    if to_radians:
        angle = np.deg2rad(angle)

    c, s = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[c, -s], [s,  c]])
    origin = np.atleast_2d(origin)
    x = np.atleast_2d(x)
    return np.squeeze((rotation_matrix @ (x.T - origin.T) + origin.T).T)


def contains(a: Union[list, tuple, np.ndarray],
             b: Union[list, tuple, np.ndarray],
             axis: int = -1,
             reduce: bool = True) -> Union[bool, np.ndarray]:
    """ Checks, whether `a` contains (elements of) `b` along a specified `axis`

    :param a: Matrix or list of vectors that is tested to contain `b`.
    :param b: List or vector that may be contained in `a`.
    :param axis: Integer specifying which `axis` of `a` is scanned for occurrences of `b`, defaults to `-1`.
    :param reduce: Boolean specifying, when `True`, whether to map any occurrence (entire absense) of `b` in `a`
                   to the return code `True` (`False`), or whether to return the element wise comparison.
    :return: `True` (`False`) if `b` occurs at any point (no where) in `a` when `reduce==True`, otherwise when
             `reduce==True` a boolean `np.array` of the element wise comparison of `b` with `a` along the
             specified `axis`.
    """
    if a is None or not len(a) or not len(b):
        return False

    a = np.atleast_2d(a)
    overlap = ~np.any(np.asarray(a) - b, axis=axis)
    if not reduce:
        return overlap
    return np.any(overlap)


def map_to_closest(a, b, indices=False, axis=1, **kwargs):
    """ Map elements of `a` to closest occurrence in `b` along `axis`
        with respect to the `np.linalg.norm` distance metric.

    :param a: Vector or List of Vector which are element-wise tested for closest occurrence in array `b`  along `axis`.
    :param b: Array which is searched along `axis` for occurrence of element in `a`.
    :param indices: Boolean specifier whether to return the `b`-indices of `a`-occurrences (if `True`),
                    or the `b` elements of `a` occurrences otherwise (if `False`), defaults to `False`.
    :param axis: Dimension along which to scan `b` for `a` elements, defaults to `1`.
    :param kwargs: (Optional) Keyword arguments for `np.linalg.norm` distance metric,
                   which is used to compare `a` with elements of `b`.
    :return: Array of (indices of) closest element in `b` to elements in `a` along `b`-`axis`,
             whether `indices == (False) True`.
    """
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    if indices:
        return np.apply_along_axis(lambda x_i: np.argmin(np.linalg.norm(b - x_i, axis=-1, **kwargs)), arr=a, axis=axis)

    return np.apply_along_axis(lambda x_i: b[np.argmin(np.linalg.norm(b - x_i, axis=-1, **kwargs))], arr=a, axis=axis)
