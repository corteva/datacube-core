# TODO: needs an aggregation phase (use xarray.DataArray.groupby?)
# TODO: collate index_measurement
# TODO: measurement dependency tracking
# TODO: a mechanism for per-leaf settings
# TODO: what does GridWorkflow do more than this?
"""
Implementation of virtual products. Provides an interface for the products in the datacube
to query and to load data, and combinators to combine multiple products into "virtual"
products implementing the same interface.
"""
from abc import ABC, abstractmethod
from functools import reduce

import xarray

from datacube import Datacube
from datacube.model import Measurement
from datacube.model.utils import xr_apply
from datacube.api.query import Query, query_group_by, query_geopolygon
from datacube.api.grid_workflow import _fast_slice
from datacube.api.core import select_datasets_inside_polygon, output_geobox


class VirtualProductException(Exception):
    """ Raised if the construction of the virtual product cannot be validated. """
    pass


class VirtualProduct(ABC):
    """ Abstract class defining the common interface of virtual products. """

    @abstractmethod
    def output_measurements(self, product_definitions):
        # type: (Dict[str, Dict]) -> Dict[str, Measurement]
        """
        A dictionary mapping names to measurement metadata.
        :param product_definitions: a dictionary mapping product names to definitions
        """

    @abstractmethod
    def find_datasets(self, dc, **query):
        # type: (Datacube, Dict[str, Any]) -> DatasetPile
        """ Collection of datasets that match the query. """

    # no index access below this line

    @abstractmethod
    def group_datasets(self, datasets, **query):
        # type: (DatasetPile, Dict[str, Any]) -> GroupedDatasetPile
        """
        Datasets grouped by their timestamps.
        :param datasets: the `DatasetPile` to fetch data from
        :param query: to specify a spatial sub-region
        """

    @abstractmethod
    def fetch_data(self, grouped, product_definitions):
        # type: (GroupedDatasetPile, Dict[str, Dict]) -> xarray.Dataset
        """ Convert grouped datasets to `xarray.Dataset`. """

    def load(self, dc, **query):
        # type: (Datacube, Dict[str, Any]) -> xarray.Dataset
        """ Mimic `datacube.Datacube.load`. """
        product_definitions = product_definitions_from_index(dc.index)
        datasets = self.find_datasets(dc, **query)
        grouped = self.group_datasets(datasets, **query)

        # for now, fetch one observation at a time
        observations = [self.fetch_data(observation, product_definitions)
                        for observation in grouped.split(dim='time')]
        data = xarray.concat(observations, dim='time')

        return data


class DatasetPile(object):
    """ Result of `VirtualProduct.find_datasets`. """
    def __init__(self, kind, pile, grid_spec):
        assert kind in ['basic', 'collate', 'juxtapose']
        self.kind = kind
        self.pile = tuple(pile)
        self.grid_spec = grid_spec


class GroupedDatasetPile(object):
    """ Result of `VirtualProduct.group_datasets`. """
    # our replacement for grid_workflow.Tile basically
    # TODO: copy the Tile API
    def __init__(self, pile, geobox):
        self.pile = pile
        self.geobox = geobox

    @property
    def dims(self):
        """
        Names of the dimensions, e.g., ``('time', 'y', 'x')``.
        :return: tuple(str)
        """
        return self.pile.dims + self.geobox.dimensions

    @property
    def shape(self):
        """
        Lengths of each dimension, e.g., ``(285, 4000, 4000)``.
        :return: tuple(int)
        """
        return self.pile.shape + self.geobox.shape

    def __getitem__(self, chunk):
        pile = self.pile

        return GroupedDatasetPile(_fast_slice(pile, chunk[:len(pile.shape)]),
                                  self.geobox[chunk[len(pile.shape):]])

    def map(self, func, dtype='O'):
        return GroupedDatasetPile(xr_apply(self.pile, func, dtype=dtype), self.geobox)

    def filter(self, predicate):
        mask = self.map(predicate, dtype='bool')

        # NOTE: this could possibly result in an empty pile
        return GroupedDatasetPile(self.pile[mask.pile], self.geobox)

    def split(self, dim='time'):
        # this is slightly different from Tile.split
        pile = self.pile

        [length] = pile[dim].shape
        for i in range(length):
            yield GroupedDatasetPile(pile.isel(**{dim: slice(i, i + 1)}), self.geobox)


class BasicProduct(VirtualProduct):
    """ A product already in the datacube. """
    def __init__(self, product_name, measurement_names=None,
                 source_filter=None, fuse_func=None, resampling_method=None,
                 dataset_filter=None):
        """
        :param product_name: name of the product
        :param measurement_names: list of names of measurements to include (None if all)
        :param source_filter: as in `Datacube.load`
        :param fuse_func: to de-duplicate
        :param resampling_method: a resampling method that applies to all measurements
        :param dataset_filter: a predicate on `datacube.Dataset` objects
        """
        # NOTE: if group_by solar_day is implemented as a transform
        #       fuse_func should never be used here
        # NOTE: resampling_method can easily be a per-measurement setting

        self.product_name = product_name

        if measurement_names is not None and len(measurement_names) == 0:
            raise VirtualProductException("Product selects no measurements")

        self.measurement_names = measurement_names

        # is this a good place for these?
        self.source_filter = source_filter
        self.fuse_func = fuse_func
        self.resampling_method = resampling_method

        self.dataset_filter = dataset_filter

    def output_measurements(self, product_definitions):
        """ Output measurements metadata. """
        measurement_docs = product_definitions[self.product_name]['measurements']
        measurements = {measurement['name']: Measurement(**measurement)
                        for measurement in measurement_docs}

        if self.measurement_names is None:
            return measurements

        try:
            return {name: measurements[name] for name in self.measurement_names}
        except KeyError as ke:
            raise VirtualProductException("Could not find measurement: {}".format(ke.args))

    def find_datasets(self, dc, **query):
        # this is basically a copy of `datacube.Datacube.find_datasets_lazy`
        # ideally that method would look like this too in the future

        # `like` is implicitly supported here
        # `platform` and `product_type` based queries are possibly ruled out
        # other possible query entries include `geopolygon`
        # and contents of `SPATIAL_KEYS` and `CRS_KEYS`
        # query should not include contents of `OTHER_KEYS` except `geopolygon`
        index = dc.index

        # find the datasets

        # Q: are measurements ever used to find datasets?
        query = Query(index, product=self.product_name, measurements=self.measurement_names,
                      source_filter=self.source_filter, **query)
        assert query.product == self.product_name

        datasets = select_datasets_inside_polygon(index.datasets.search(**query.search_terms),
                                                  query.geopolygon)

        if self.dataset_filter is not None:
            datasets = [dataset for dataset in datasets if self.dataset_filter(dataset)]

        # gather information from the index before it disappears from sight
        # this can also possibly extracted from the product definitions but this is easier
        grid_spec = index.products.get_by_name(self.product_name).grid_spec

        return DatasetPile('basic', datasets, grid_spec)

    def group_datasets(self, datasets, **query):
        assert isinstance(datasets, DatasetPile) and datasets.kind == 'basic'
        pile = datasets.pile
        grid_spec = datasets.grid_spec

        # we will support group_by='solar_day' elsewhere
        assert 'group_by' not in query

        # possible query entries are contents of `SPATIAL_KEYS`, `CRS_KEYS`, and `OTHER_KEYS`
        # query should not include `product`, `measurements`, and `resampling`

        # select only those inside the ROI
        # ROI could be smaller than the query for `find_datasets`
        polygon = query_geopolygon(**query)
        selected = list(select_datasets_inside_polygon(pile, polygon))

        # geobox
        geobox = output_geobox(datasets=pile, grid_spec=grid_spec, **query)

        # group by time
        grouped = Datacube.group_datasets(selected, query_group_by(group_by='time'))

        def wrap(_, value):
            return DatasetPile('basic', value, grid_spec)

        # information needed for Datacube.load_data
        return GroupedDatasetPile(grouped, geobox).map(wrap)

    def fetch_data(self, grouped, product_definitions):
        assert isinstance(grouped, GroupedDatasetPile)

        # this method is basically `GridWorkflow.load`

        # essentially what `datacube.api.core.set_resampling_method` does
        if self.resampling_method is not None:
            resampling = {'resampling_method': self.resampling_method}
        else:
            resampling = {}

        measurements = [measurement.to_dict(update_with=resampling)
                        for _, measurement in self.output_measurements(product_definitions).items()]

        def unwrap(_, value):
            assert isinstance(value, DatasetPile) and value.kind == 'basic'
            return value.pile

        return Datacube.load_data(grouped.map(unwrap).pile,
                                  grouped.geobox, measurements, fuse_func=self.fuse_func)


def basic_product(product_name, measurement_names=None,
                  source_filter=None, fuse_func=None, resampling_method=None):
    return BasicProduct(product_name, measurement_names=measurement_names,
                        source_filter=source_filter, fuse_func=fuse_func,
                        resampling_method=resampling_method)


class Transform(VirtualProduct):
    """
    Apply some computation to the loaded data.
    """
    def __init__(self, child,
                 data_transform=None, measurement_transform=None,
                 query_transform=None):
        self.child = child

        def identity(x):
            return x

        def guard(func):
            if func is None:
                return identity
            return func

        self.data_transform = guard(data_transform)
        self.measurement_transform = guard(measurement_transform)
        self.query_transform = guard(query_transform)

    def output_measurements(self, product_definitions):
        return self.measurement_transform(self.child.output_measurements(product_definitions))

    def find_datasets(self, dc, **query):
        return self.child.find_datasets(dc, **self.query_transform(query))

    def group_datasets(self, datasets, **query):
        return self.child.group_datasets(datasets, **self.query_transform(query))

    def fetch_data(self, grouped, product_definitions):
        return self.data_transform(self.child.fetch_data(grouped, product_definitions))


def transform(child, data_transform=None, measurement_transform=None,
              query_transform=None):
    return Transform(child, data_transform=data_transform,
                     measurement_transform=measurement_transform,
                     query_transform=query_transform)


class Collate(VirtualProduct):
    def __init__(self, *children, index_measurement_name=None):
        if len(children) == 0:
            raise VirtualProductException("No children for collate node")

        self.children = children
        self.index_measurement_name = index_measurement_name

        name = self.index_measurement_name
        if name is not None:
            self.index_measurement = {
                name: Measurement(name=name, dtype='int8', nodata=-1, units='1')
            }

    def output_measurements(self, product_definitions):
        input_measurements = [child.output_measurements(product_definitions)
                              for child in self.children]

        first, *rest = input_measurements

        for child in rest:
            if set(child) != set(first):
                msg = "Child datasets do not all have the same set of measurements"
                raise VirtualProductException(msg)

        if self.index_measurement_name is None:
            return first

        if self.index_measurement_name in first:
            msg = "Source index measurement '{}' already present".format(self.index_measurement_name)
            raise VirtualProductException(msg)

        return {**first, **self.index_measurement}

    def find_datasets(self, dc, **query):
        result = [child.find_datasets(dc, **query)
                  for child in self.children]

        grid_spec = select_unique([datasets.grid_spec for datasets in result])
        return DatasetPile('collate', result, grid_spec)

    def group_datasets(self, datasets, **query):
        assert isinstance(datasets, DatasetPile) and datasets.kind == 'collate'
        assert len(datasets.pile) == len(self.children)
        grid_spec = datasets.grid_spec

        def build(source_index, product, dataset_pile):
            grouped = product.group_datasets(dataset_pile, **query)

            def tag(_, value):
                in_position = [value if i == source_index else None
                               for i, _ in enumerate(self.children)]
                return DatasetPile('collate', in_position, grid_spec)

            return grouped.map(tag)

        groups = [build(source_index, product, dataset_pile)
                  for source_index, (product, dataset_pile)
                  in enumerate(zip(self.children, datasets.pile))]

        geobox = select_unique([grouped.geobox for grouped in groups])

        concatenated = xarray.concat([grouped.pile for grouped in groups], dim='time')
        return GroupedDatasetPile(concatenated, geobox)

    def fetch_data(self, grouped, product_definitions):
        assert isinstance(grouped, GroupedDatasetPile)

        def is_from(source_index):
            def result(_, value):
                assert isinstance(value, DatasetPile) and value.kind == 'collate'
                return value.pile[source_index] is not None

            return result

        def strip_source(_, value):
            assert isinstance(value, DatasetPile) and value.kind == 'collate'
            for data in value.pile:
                if data is not None:
                    return data

            raise ValueError("Every child of CollatedDatasetPile object is None")

        def fetch_child(child, r):
            size = reduce(lambda x, y: x * y, r.shape, 1)

            if size > 0:
                # TODO: merge with source_index here
                # requires passing source_index to this
                return child.fetch_data(r, product_definitions)
            else:
                # empty raster
                return None

        groups = [fetch_child(child, grouped.filter(is_from(source_index)).map(strip_source))
                  for source_index, child in enumerate(self.children)]

        non_empty = [g for g in groups if g is not None]
        attrs = select_unique([g.attrs for g in non_empty])

        return xarray.concat(non_empty, dim='time').assign_attrs(**attrs)


def collate(*children, index_measurement_name=None):
    return Collate(*children, index_measurement_name=index_measurement_name)


class Juxtapose(VirtualProduct):
    def __init__(self, *children):
        if len(children) == 0:
            raise VirtualProductException("No children for juxtapose node")

        self.children = children

    def output_measurements(self, product_definitions):
        input_measurements = [child.output_measurements(product_definitions)
                              for child in self.children]

        result = {}
        for measurements in input_measurements:
            common = set(result) & set(measurements)
            if common != set():
                msg = "Common measurements {} between children".format(common)
                raise VirtualProductException(msg)

            result.update(measurements)

        return result

    def find_datasets(self, dc, **query):
        result = [child.find_datasets(dc, **query) for child in self.children]

        grid_spec = select_unique([datasets.grid_spec for datasets in result])
        return DatasetPile('juxtapose', result, grid_spec)

    def group_datasets(self, datasets, **query):
        assert isinstance(datasets, DatasetPile) and datasets.kind == 'juxtapose'
        assert len(datasets.pile) == len(self.children)

        pile = datasets.pile
        grid_spec = datasets.grid_spec

        groups = [product.group_datasets(datasets, **query)
                  for product, datasets in zip(self.children, pile)]

        geobox = select_unique([grouped.geobox for grouped in groups])

        aligned_piles = xarray.align(*[grouped.pile for grouped in groups])
        child_groups = [GroupedDatasetPile(aligned_piles[i], grouped.geobox)
                        for i, grouped in enumerate(groups)]

        def tuplify(indexes, _):
            return DatasetPile('juxtapose',
                               [grouped.pile.sel(**indexes).item() for grouped in child_groups], grid_spec)

        merged = child_groups[0].map(tuplify).pile

        return GroupedDatasetPile(merged, geobox)

    def fetch_data(self, grouped, product_definitions):
        assert isinstance(grouped, GroupedDatasetPile)
        geobox = grouped.geobox

        def select_child(source_index):
            def result(_, value):
                assert isinstance(value, DatasetPile) and value.kind == 'juxtapose'
                return value.pile[source_index]

            return result

        def fetch_recipe(source_index):
            child_groups = grouped.map(select_child(source_index))
            return GroupedDatasetPile(child_groups.pile, geobox)

        groups = [child.fetch_data(fetch_recipe(source_index), product_definitions)
                  for source_index, child in enumerate(self.children)]

        attrs = select_unique([g.attrs for g in groups])
        return xarray.merge(groups).assign_attrs(**attrs)


def juxtapose(*children):
    return Juxtapose(*children)


def product_definitions_from_index(index):
    return {product.name: product.definition
            for product in index.products.get_all()}


def select_unique(things):
    """ Checks that all the members of `things` are equal, and then returns it. """
    first, *rest = things
    for other in rest:
        # should possibly just emit a warning
        assert first == other, "{} is not the same as {}".format(first, other)

    return first
