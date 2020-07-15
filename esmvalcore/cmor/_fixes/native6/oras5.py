"""Fixes for ERA5."""
import datetime
import logging

import iris
import numpy as np

from ..fix import Fix
from ..shared import cube_to_aux_coord

logger = logging.getLogger(__name__)

class Thetao(Fix):
    """Fixes for Geopotential."""
    def fix_metadata(self, cubes):
        """Fix metadata."""
        cube = cubes.extract('votemper')[0]
        cube.standard_name = self.vardef.standard_name
        cube.long_name = self.vardef.long_name
        cube.var_name = self.vardef.short_name
        cube.units = self.vardef.units
        levels = cube.coord(var_name='deptht')
        try:
            cube.coord(var_name='time')
        except iris.exceptions.CoordinateNotFoundError:
            cube.add_dim_coord(
                iris.coords.DimCoord(
                    points=cubes.extract('time')[0].core_data(),
                    var_name='time',
                    standard_name='time',
                    units=cubes.extract('time')[0].units,
                    ),
                0)

        for coord in cube.coords():
            if coord.var_name != 'time':
                cube.remove_coord(coord)
        cube.add_aux_coord(
            cube_to_aux_coord(
                cubes.extract('latitude')[0]), (2, 3))
        cube.add_aux_coord(
            cube_to_aux_coord(
                cubes.extract('longitude')[0]), (2, 3))
        cube.add_dim_coord(
            iris.coords.DimCoord(
                points=levels.points,
                var_name='lev',
                standard_name='depth',
                long_name='ocean depth coordinate',
                units='m',
                attributes={'positive': 'down'}
                ),
            1)
        return cube

class So(Fix):
    """Fixes for Geopotential."""
    def fix_metadata(self, cubes):
        """Fix metadata."""
        cube = cubes.extract('vosaline')[0]
        cube.standard_name = self.vardef.standard_name
        cube.long_name = self.vardef.long_name
        cube.var_name = self.vardef.short_name
        cube.units = self.vardef.units
        levels = cube.coord(var_name='deptht')
        try:
            cube.coord(var_name='time')
        except iris.exceptions.CoordinateNotFoundError:
            cube.add_dim_coord(
                iris.coords.DimCoord(
                    points=cubes.extract('time')[0].core_data(),
                    var_name='time',
                    standard_name='time',
                    units=cubes.extract('time')[0].units,
                    ),
                0)

        for coord in cube.coords():
            if coord.var_name != 'time':
                cube.remove_coord(coord)
        cube.add_aux_coord(
            cube_to_aux_coord(
                cubes.extract('latitude')[0]), (2, 3))
        cube.add_aux_coord(
            cube_to_aux_coord(
                cubes.extract('longitude')[0]), (2, 3))
        cube.add_dim_coord(
            iris.coords.DimCoord(
                points=levels.points,
                var_name='lev',
                standard_name='depth',
                long_name='ocean depth coordinate',
                units='m',
                ),
            1)
        del cube.attributes['invalid_units']
        return cube

class AllVars(Fix):
    """Fixes for all variables."""
    def fix_metadata(self, cubes):
        """Fix metadata."""
        fixed_cubes = iris.cube.CubeList()
        i = iris.coords.DimCoord(
            np.arange(1, cubes.shape[3]+1).astype(np.int32),
            long_name='cell index along first dimension',
            units='1',
            var_name='i')
        j = iris.coords.DimCoord(
            np.arange(1, cubes.shape[2]+1).astype(np.int32),
            long_name='cell index along second dimension',
            units='1',
            var_name='j')
        cubes.add_dim_coord(j, 2)
        cubes.add_dim_coord(i, 3)
        f = np.vectorize(lambda x: x % 360)
        cubes.coord('longitude').points = f(cubes.coord('longitude').points)
        for name in ['latitude', 'longitude']:
            cubes.coord(name).units = self.vardef.coordinates[name].units
            cubes.coord(name).bounds = self.create_bounds(cubes.coord(name).points, name)
        fixed_cubes.append(cubes)
        return fixed_cubes
    
    def create_bounds(self, coord, name):
        if name == 'latitude':
            return self.create_vertex_lats(coord)
        else:
            return self.create_vertex_lons(coord)
    
    def create_vertex_lons(self, a):
        ny = a.shape[0]
        nx = a.shape[1]
        f = np.vectorize(lambda x: x % 360)
        if nx == 1:  # Longitudes were integrated out
            if ny == 1:
                return f(np.array([a[0, 0]]))
            return np.zeros([ny, 2])
        b = np.zeros([ny, nx, 4])
        b[:, 1:nx, 0] = f(0.5 * (a[:, 0:nx - 1] + a[:, 1:nx]))
        b[:, 0, 0] = f(1.5 * a[:, 0] - 0.5 * a[:, 1])
        b[:, 0:nx - 1, 1] = b[:, 1:nx, 0]
        b[:, nx - 1, 1] = f(1.5 * a[:, nx - 1] - 0.5 * a[:, nx - 2])
        b[:, :, 2] = b[:, :, 1]
        b[:, :, 3] = b[:, :, 0]
        return b
    
    def create_vertex_lats(self, a):
        ny = a.shape[0]
        nx = a.shape[1]
        f = np.vectorize(lambda x: (x + 90) % 180 - 90)
        if nx == 1:  # Longitudes were integrated out
            if ny == 1:
                return f(np.array([a[0, 0]]))
            b = np.zeros([ny, 2])
            b[1:ny, 0] = f(0.5 * (a[0:ny - 1, 0] + a[1:ny, 0]))
            b[0, 0] = f(2 * a[0, 0] - b[1, 0])
            b[0:ny - 1, 1] = b[1:ny, 0]
            b[ny - 1, 1] = f(1.5 * a[ny - 1, 0] - 0.5 * a[ny - 2, 0])
            return b
        b = np.zeros([ny, nx, 4])
        b[1:ny, :, 0] = f(0.5 * (a[0:ny - 1, :] + a[1:ny, :]))
        b[0, :, 0] = f(2 * a[0, :] - b[1, :, 0])
        b[:, :, 1] = b[:, :, 0]
        b[0:ny - 1, :, 2] = b[1:ny, :, 0]
        b[ny - 1, :, 2] = f(1.5 * a[ny - 1, :] - 0.5 * a[ny - 2, :])
        b[:, :, 3] = b[:, :, 2]
        return b

