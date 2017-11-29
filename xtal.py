from scipy.spatial import Voronoi, ConvexHull
import numpy as np
import flexxtools as ft
import json


class BZ3D(object):
    @property
    def space_group(self):
        return self._space_group

    @space_group.setter
    def space_group(self, value):
        if value not in symmetry_operations_dict.keys() and value is not None:
            raise IndexError('Attempted to set invalid space group to Brillouin zone object.')
        else:
            self._space_group = value
            self._symops = symmetry_operations_dict[value]['ops']
            self.reflections_valid = False
            self._2dbz = None
            self._2dbz_p = None

    @property
    def reflections(self):
        if self.reflections_valid:
            return self._reflections
        else:
            raise RuntimeError('BZ not yet calculated and not ready for querying.')

    def __init__(self, ub_matrix: ft.UBMatrix = None, central_ref=(0, 0, 0), space_group='unset'):
        self._symops = []
        self._rec_cell_polys = None
        self._2dbz = None
        self._2dbz_p = None
        self._space_group = None
        self.reflections_valid = False
        self.space_group = space_group


        self._central_ref = None
        self.ub_matrix = ub_matrix
        self._reflections = []

    def calcBZ(self, normal=(0, 0, 1)):
        # result stored in S coordinates
        self.generate_reflections()
        r_coords = np.array(self._reflections).T
        s_coords = self.ub_matrix.convert(r_coords, 'rs')
        vor = Voronoi(s_coords.T)

        vertices = []
        for vert_index in vor.regions[vor.point_region[0]]:
            vertices.append(vor.vertices[vert_index])

        hull = ConvexHull(vertices)
        polygons = []
        for polygon in hull.simplices:
            polygon_vertices = [hull.points[i] for i in polygon]
            polygons.append(polygon_vertices)

        self._rec_cell_polys = polygons

    def first_bz_intersection(self):
        points_in_s = []
        for verts in self._rec_cell_polys:
            for i in range(-1, 2):
                vert_1 = verts[i]
                vert_2 = verts[i+1]
                if vert_2[2] == 0:
                    points_in_s.append(tuple(vert_2))
                z1, z2 = (vert_1[2], vert_2[2])
                if z1 * z2 < 0:
                    ratio = (abs(z2) / (abs(z1) + abs(z2)), abs(z1) / (abs(z1) + abs(z2)))
                    points_in_s.append(tuple(vert_1 * ratio[0] + vert_2 * ratio[1]))

        points_unique_array = np.array(list(set(points_in_s)))
        hull_2d = ConvexHull(points_unique_array[:, 0:2])
        verts_2d = np.array([hull_2d.points[vert_id] for vert_id in hull_2d.vertices])
        verts_2d_3col = np.hstack((verts_2d, np.zeros([verts_2d.shape[0], 1])))
        verts_2d_p = self.ub_matrix.convert(verts_2d_3col.T, 'sp')
        verts_2d_p_closed = np.hstack((verts_2d_p, verts_2d_p[:, -1]))
        self._2dbz_p = verts_2d_p


    def generate_reflections(self):
        self._reflections = [[0, 0, 0]]

        for h in range(-4, 5):
            for k in range(-4, 5):
                for l in range(-4, 5):
                    if not check_forbidden([h, k, l], self._symops) and not (h == 0 and k == 0 and l == 0):
                        self._reflections.append([h, k, l])

        self.reflections_valid = True

    def check_forbidden(self, hkl):
        return check_forbidden(hkl, self._symops)


def check_forbidden(hkl, sg_or_oplist):
    const = np.multiply(2 * np.pi, np.complex(0, 1))
    hkl = np.array(hkl)
    if type(sg_or_oplist) is not str:
        ops = sg_or_oplist
    else:
        try:
            ops = symmetry_operations_dict[sg_or_oplist]['ops']
        except KeyError:
            print('failed to retrieve space group %s:' % sg_or_oplist)
            return False

    for op in ops:
        m, v = parse_sym_op(op)
        if np.linalg.norm(v) == 0:
            continue
        if np.all(np.dot(m, hkl) == hkl):
            if not np.isclose(np.exp(np.multiply(const, np.dot(hkl, v))).real, 1) and \
                    np.isclose(np.exp(np.multiply(const, np.dot(hkl, v))).imag, 0):
                return True

    return False


def load_space_groups(file='symops.json'):
    return json.loads(open(file).read())


def parse_sym_op(operation):
    matrix = np.array([[operation[0], operation[1], operation[2]], [operation[3], operation[4], operation[5]],
                       [operation[6], operation[7], operation[8]]]).T
    vector = np.array([operation[9], operation[10], operation[11]])
    return matrix, vector


symmetry_operations_dict = load_space_groups()
