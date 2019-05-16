"""ESMValTool CMORizer for CRU data.

Tier
    Tier 2: other freely-available dataset.

Source
    https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_4.02/cruts.1811131722.v4.02/

Last access
    20190516

Download and processing instructions
    Download the following files:
        cru_ts4.02.1901.2017.{raw_name}.dat.nc
    where {raw_name} is the name of the desired variable.

"""

import logging
import os

import iris

import esmvaltool.utils.cmorizers.obs.utilities as utils

logger = logging.getLogger(__name__)

CFG = utils.read_cmor_config('CRU.yml')


def _extract_variable(raw_var, cmor_info, attrs, filepath, out_dir):
    """Extract variable."""
    var = cmor_info.short_name
    cube = iris.load_cube(filepath, utils.var_name_constraint(raw_var))
    utils.fix_var_metadata(cube, cmor_info)
    utils.convert_timeunits(cube, 1950)
    utils.fix_coords(cube)
    utils.set_global_atts(cube, attrs)
    if var in ('tas',):
        utils.add_height2m(cube)
    utils.save_variable(
        cube, var, out_dir, attrs, unlimited_dimensions=['time'])


def cmorization(in_dir, out_dir):
    """Cmorization func call."""
    glob_attrs = CFG['attributes']
    cmor_table = CFG['cmor_table']
    logger.info("Starting cmorization for Tier%s OBS files: %s",
                glob_attrs['tier'], glob_attrs['dataset_id'])
    logger.info("Input data from: %s", in_dir)
    logger.info("Output will be written to: %s", out_dir)
    raw_filepath = os.path.join(in_dir, CFG['filename'])

    # Run the cmorization
    for (var, var_info) in CFG['variables'].items():
        logger.info("CMORizing variable '%s'", var)
        glob_attrs['mip'] = var_info['mip']
        cmor_info = cmor_table.get_variable(var_info['mip'], var)
        raw_var = var_info.get('raw', var)
        filepath = raw_filepath.format(raw_name=raw_var)
        if not os.path.isfile(filepath):
            logger.debug("Skipping '%s', file '%s' not found", var, filepath)
            continue
        logger.info("Found input file '%s'", filepath)
        _extract_variable(raw_var, cmor_info, glob_attrs, filepath, out_dir)
