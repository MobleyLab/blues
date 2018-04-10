# -*- coding: utf-8 -*-
from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

from chemcoord.exceptions import ERR_CODE_OK, \
    InvalidReference, ERR_CODE_InvalidReference
import chemcoord.constants as constants
from chemcoord.cartesian_coordinates.xyz_functions import \
    _jit_normalize, \
    _jit_get_rotation_matrix, \
    _jit_isclose, \
    _jit_cross
from numba import jit
import numba as nb
import numpy as np
import pandas as pd

#@jit(nopython=True)
def _jit_calc_single_position_edit(references, zmat_values, row):
    bond, angle, dihedral = zmat_values[row]
    vb, va, vd = references[0], references[1], references[2]
    #zeros = np.zeros(3, dtype=nb.types.f8)
    zeros = np.zeros(3)


    BA = va - vb
    if _jit_isclose(BA, zeros).all():
        return (ERR_CODE_InvalidReference, zeros)
    ba = _jit_normalize(BA)
    if _jit_isclose(angle, np.pi):
        d = bond * -ba
    elif _jit_isclose(angle, 0.):
        d = bond * ba
    else:
        AD = vd - va
        N1 = _jit_cross(BA, AD)
        if _jit_isclose(N1, zeros).all():
            return (ERR_CODE_InvalidReference, zeros)
        else:
            n1 = _jit_normalize(N1)
            d = bond * ba
            d = np.dot(_jit_get_rotation_matrix(n1, angle), d)
            d = np.dot(_jit_get_rotation_matrix(ba, dihedral), d)
    return (ERR_CODE_OK, vb + d)

#@jit(nopython=True)
def _jit_calc_positions_edit(c_table, zmat_values, start_coord):
    edit_coord = start_coord
    #edit_coord = start_coord.astype(dtype=nb.types.f8)


    n_atoms = c_table.shape[0]
    #positions = np.empty((n_atoms, 3), dtype=nb.types.f8)
    positions = np.empty((n_atoms, 3))
    for row in range(n_atoms):
        ref_pos = np.empty((3, 3))
        if row <= 2:
            positions[row] = edit_coord[row]
        else:
            for k in range(3):
                j = c_table[row, k]
                if j < constants.keys_below_are_abs_refs:
                    ref_pos[k] = constants._jit_absolute_refs(j)
                else:
                    ref_pos[k] = positions[j]
            err, pos = _jit_calc_single_position_edit(ref_pos, zmat_values, row)
            if err == ERR_CODE_OK:
                positions[row] = pos
            else:
                return (err, row, positions)

    return (ERR_CODE_OK, row, positions)


def give_cartesian_edit(self, start_coord):
    """sets the  cartexian coordinates of the first 3 atoms and then moves the internal
    coordinates into carteasian space with those first 3 to specifiy the absolute orientation

    Parameters
    ----------
    start_coord: 3x3 numpy.array

    Returns
    -------
    cartesian: Chemcoord.Cartesian object
    """
    zmat = self.change_numbering()
    c_table = zmat.loc[:, ['b', 'a', 'd']].values
    zmat_values = zmat.loc[:, ['bond', 'angle', 'dihedral']].values
    zmat_values[:, [1, 2]] = np.radians(zmat_values[:, [1, 2]])

    def create_cartesian(positions, row):
        xyz_frame = pd.DataFrame(columns=['atom', 'x', 'y', 'z'],
                                 index=self.index[:row], dtype='f8')
        xyz_frame['atom'] = self.loc[xyz_frame.index, 'atom']
        xyz_frame.loc[:, ['x', 'y', 'z']] = positions[:row]
        from chemcoord.cartesian_coordinates.cartesian_class_main \
            import Cartesian
        cartesian = Cartesian(xyz_frame)
        return cartesian

    err, row, positions = _jit_calc_positions_edit(c_table, zmat_values, start_coord)

    if err == ERR_CODE_InvalidReference:
        rename = dict(enumerate(self.index))
        i = rename[row]
        b, a, d = self.loc[i, ['b', 'a', 'd']]
        cartesian = create_cartesian(positions, row)
        raise InvalidReference(i=i, b=b, a=a, d=d,
                               already_built_cartesian=cartesian)
    elif err == ERR_CODE_OK:
        return create_cartesian(positions, row + 1)



