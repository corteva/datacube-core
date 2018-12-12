""" Dataset -> Raster
"""
from affine import Affine
import numpy as np

from datacube.utils.geometry import (
    roi_shape,
    GeoBox,
    w_,
    warp_affine,
    rio_reproject,
    compute_reproject_roi)

from datacube.utils.geometry import gbox as gbx


def rdr_geobox(rdr):
    """ Construct GeoBox from opened dataset reader.
    """
    h, w = rdr.shape
    return GeoBox(w, h, rdr.transform, rdr.crs)


def is_almost_int(x, tol):
    from math import fmod

    x = abs(fmod(x, 1))
    if x > 0.5:
        x = 1 - x
    return x < tol


def can_paste(rr, stol=1e-3, ttol=1e-2):
    """
    Take result of compute_reproject_roi and check if can read(possibly with scale) and paste,
    or do we need to read then reproject.

    :returns: (True, None) if one can just read and paste
    :returns: (False, Reason) if pasting is not possible, so need to reproject after reading
    """
    if not rr.is_st:  # not linear or not Scale + Translation
        return False, "not ST"

    scale = rr.scale
    if not is_almost_int(scale, stol):  # non-integer scaling
        return False, "non-integer scale"

    scale = np.round(scale)
    A = rr.transform.linear           # src -> dst
    A = A*Affine.scale(scale, scale)  # src.overview[scale] -> dst

    (sx, _, tx,  # tx, ty are in dst pixel space
     _, sy, ty,
     *_) = A

    if any(abs(abs(s) - 1) > stol
           for s in (sx, sy)):  # not equal scaling across axis?
        return False, "sx!=sy, probably"

    ny, nx = (n/scale
              for n in roi_shape(rr.roi_src))

    # src_roi doesn't divide by scale properly:
    #  example 3x7 scaled down by factor of 2
    if not all(is_almost_int(n, stol) for n in (nx, ny)):
        return False, "src_roi doesn't align for scale"

    # scaled down shape doesn't match dst shape
    s_shape = (int(ny), int(nx))
    if s_shape != roi_shape(rr.roi_dst):
        return False, "src_roi/scale != dst_roi"

    # final check: sub-pixel translation
    if not all(is_almost_int(t, ttol) for t in (tx, ty)):
        return False, "sub-pixel translation"

    return True, None


def valid_mask(xx, nodata):
    if np.isnan(nodata):
        return ~np.isnan(xx)
    if nodata is None:
        return np.ones(xx.shape, dtype='bool')
    return xx != nodata


def pick_read_scale(scale: float, rdr=None, tol=1e-3):
    assert scale > 0
    # First find nearest integer scale
    #    Scale down to nearest integer, unless we can scale up by less than tol
    #
    # 2.999999 -> 3
    # 2.8 -> 2
    # 0.3 -> 1

    if scale < 1:
        return 1

    if is_almost_int(scale, tol):
        scale = np.round(scale)

    scale = int(scale)

    if rdr is not None:
        # TODO: check available overviews in rdr
        pass

    return scale


def read_time_slice(rdr, dst, dst_gbox, resampling, dst_nodata):
    assert dst.shape == dst_gbox.shape
    src_gbox = rdr_geobox(rdr)

    rr = compute_reproject_roi(src_gbox, dst_gbox)
    scale = pick_read_scale(rr.scale, rdr)

    dst = dst[rr.roi_dst]
    paste_ok, _ = can_paste(rr)

    if paste_ok:
        A = rr.transform.linear
        sx, sy = A.a, A.e

        pix = rdr.read(w_[rr.roi_src], out_shape=dst.shape)

        if sx < 0:
            pix = pix[:, ::-1]
        if sy < 0:
            pix = pix[::-1, :]

        if rdr.nodata is None:
            np.copyto(dst, pix)
        else:
            np.copyto(dst, pix, where=valid_mask(pix, rdr.nodata))
    else:
        dst_gbox = dst_gbox[rr.roi_dst]
        src_gbox = src_gbox[rr.roi_src]
        if scale > 1:
            src_gbox = gbx.zoom_out(src_gbox, scale)

        pix = rdr.read(w_[rr.roi_src], out_shape=src_gbox.shape)

        if rr.transform.linear is not None:
            A = (~src_gbox.transform)*dst_gbox.transform
            warp_affine(pix, dst, A, resampling,
                        src_nodata=rdr.nodata, dst_nodata=dst_nodata)
        else:
            rio_reproject(pix, dst, src_gbox, dst_gbox, resampling,
                          src_nodata=rdr.nodata, dst_nodata=dst_nodata)

    return rr.roi_dst