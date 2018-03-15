import numpy
import xarray

from datacube.storage.masking import make_mask as make_mask_prim
from datacube.storage.masking import mask_invalid_data as mask_invalid_data_prim

from .impl import transform, VirtualProductException, Measurement


def selective_apply_dict(dictionary, apply_to=None, key_map=None, value_map=None):
    def skip(key):
        return apply_to is not None and key not in apply_to

    def key_worker(key):
        if key_map is None or skip(key):
            return key

        return key_map(key)

    def value_worker(key, value):
        if value_map is None or skip(key):
            return value

        return value_map(key, value)

    return {key_worker(key): value_worker(key, value)
            for key, value in dictionary.items()}


def selective_apply(data, apply_to=None, key_map=None, value_map=None):
    return xarray.Dataset(data_vars=selective_apply_dict(data.data_vars, apply_to=apply_to,
                                                         key_map=key_map, value_map=value_map),
                          coords=data.coords, attrs=data.attrs)


def make_mask(product, mask_measurement_name, **flags):
    """
    Create a mask that would only keep pixels for which the measurement with `mask_measurement_name`
    of the `product` satisfies `flags`.
    """
    apply_to = [mask_measurement_name]

    def data_transform(data):
        def worker(key, value):
            return make_mask_prim(value, **flags)

        return selective_apply(data, apply_to=apply_to, value_map=worker)

    def measurement_transform(input_measurements):
        if mask_measurement_name not in input_measurements:
            raise VirtualProductException("required measurement {} not found".format(mask_measurement_name))

        def worker(key, value):
            result = value.__dict__.copy()
            result['dtype'] = numpy.dtype('bool')
            return Measurement(**result)

        return selective_apply_dict(input_measurements, apply_to=apply_to, value_map=worker)

    return transform(product, data_transform=data_transform,
                     measurement_transform=measurement_transform)


def mask_by(product, mask_measurement_name, apply_to=None, preserve_dtype=True, **flags):
    """
    Only keep pixels for which the measurement with `mask_measurement_name` of the
    `product` satisfies `flags`.
    """
    def data_transform(data):
        mask = data[mask_measurement_name]
        rest = data.drop(mask_measurement_name)

        def worker(key, value):
            if preserve_dtype:
                if 'nodata' not in value.attrs:
                    raise VirtualProductException("measurement {} has no nodata value".format(key))
                return value.where(mask, value.attrs['nodata'])

            # beware! the dtype has changed to float64
            return value.where(mask)

        return selective_apply(rest, apply_to=apply_to, value_map=worker)

    def measurement_transform(input_measurements):
        rest = {key: value for key, value in input_measurements.items() if key != mask_measurement_name}

        def worker(key, value):
            if preserve_dtype:
                return value

            result = value.__dict__.copy()
            # is there a way to find this dtype dynamically?
            result['dtype'] = numpy.dtype('float64')
            return Measurement(**result)

        return selective_apply_dict(rest, apply_to=apply_to, value_map=worker)

    return transform(make_mask(product, mask_measurement_name, **flags),
                     data_transform=data_transform,
                     measurement_transform=measurement_transform)


def to_float(product, apply_to=None, dtype=numpy.dtype('float32')):
    """
    Convert the dataset to float. Replaces `nodata` values with `Nan`s.
    """
    def data_transform(data):
        def worker(key, value):
            if hasattr(value, 'dtype') and value.dtype == dtype:
                return value

            return mask_invalid_data_prim(value).astype(dtype)

        return selective_apply(data, apply_to=apply_to, value_map=worker)

    def measurement_transform(input_measurements):
        def worker(key, value):
            result = value.__dict__.copy()
            result['dtype'] = dtype
            return Measurement(**result)

        return selective_apply_dict(input_measurements, apply_to=apply_to, value_map=worker)

    return transform(product, data_transform=data_transform,
                     measurement_transform=measurement_transform)


def rename(product, name_dict):
    def data_transform(data):
        return data.rename(name_dict)

    def measurement_transform(input_measurements):
        def key_map(key):
            return name_dict[key]

        def value_map(key, value):
            result = value.__dict__.copy()
            result['name'] = name_dict[key]
            return Measurement(**result)

        return selective_apply_dict(input_measurements, apply_to=name_dict,
                                    key_map=key_map, value_map=value_map)

    return transform(product, data_transform=data_transform,
                     measurement_transform=measurement_transform)
