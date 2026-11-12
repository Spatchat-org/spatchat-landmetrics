import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np


class RecordingAxes:
    def __init__(self):
        self.legend_labels = []

    def imshow(self, _array, **_kwargs):
        return None

    def set_title(self, _title):
        return None

    def axis(self, _setting):
        return None

    def legend(self, handles, **_kwargs):
        self.legend_labels = [handle.label for handle in handles]


class FakeDataset:
    res = (1, 1)

    def __init__(self, nodata=None):
        self.nodata = nodata

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self, _band, masked=False):
        values = np.array([[0, 1]], dtype=np.uint8)
        if masked:
            mask = values == self.nodata if self.nodata is not None else False
            return np.ma.array(values, mask=mask)
        return values


def fake_rasterio(dataset):
    rasterio = types.ModuleType("rasterio")

    def open_dataset(_path):
        return dataset

    rasterio.open = open_dataset
    return rasterio


def load_module(filename, module_name, dependencies):
    module_path = Path(__file__).resolve().parents[1] / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, dependencies):
        spec.loader.exec_module(module)
    return module


def load_raster_io(dataset, landscape_class=object):
    rasterio = fake_rasterio(dataset)

    matplotlib = types.ModuleType("matplotlib")
    matplotlib.pyplot = types.ModuleType("matplotlib.pyplot")
    matplotlib.patches = types.ModuleType("matplotlib.patches")

    def subplots(**_kwargs):
        axes = RecordingAxes()
        return types.SimpleNamespace(axes=axes), axes

    def colors(values):
        return values

    def legend_patch(color, label):
        return types.SimpleNamespace(color=color, label=label)

    matplotlib.pyplot.subplots = subplots
    matplotlib.pyplot.cm = types.SimpleNamespace(tab10=colors)
    matplotlib.patches.Patch = legend_patch

    pylandstats = types.ModuleType("pylandstats")
    pylandstats.Landscape = landscape_class

    dependencies = {
        "rasterio": rasterio,
        "matplotlib": matplotlib,
        "matplotlib.pyplot": matplotlib.pyplot,
        "matplotlib.patches": matplotlib.patches,
        "gradio": types.ModuleType("gradio"),
        "pylandstats": pylandstats,
    }
    return load_module("raster_io.py", "raster_io_under_test", dependencies)


def load_data_views(dataset):
    rasterio = fake_rasterio(dataset)

    pandas = types.ModuleType("pandas")
    pandas.DataFrame = object

    matplotlib = types.ModuleType("matplotlib")
    matplotlib.use = Mock()
    matplotlib.pyplot = types.ModuleType("matplotlib.pyplot")

    scipy = types.ModuleType("scipy")
    scipy.ndimage = types.ModuleType("scipy.ndimage")

    dependencies = {
        "rasterio": rasterio,
        "pandas": pandas,
        "matplotlib": matplotlib,
        "matplotlib.pyplot": matplotlib.pyplot,
        "scipy": scipy,
        "scipy.ndimage": scipy.ndimage,
    }
    return load_module("data_views.py", "data_views_under_test", dependencies)


class CountClassesTests(unittest.TestCase):
    def test_zero_is_a_class_when_raster_has_no_nodata_value(self):
        raster_io = load_raster_io(FakeDataset())

        result = raster_io.count_classes("fixture.tif")

        self.assertEqual(result, "Your raster contains 2 unique classes.")

    def test_zero_is_excluded_when_declared_as_nodata(self):
        raster_io = load_raster_io(FakeDataset(nodata=0))

        result = raster_io.count_classes("fixture.tif")

        self.assertEqual(result, "Your raster contains 1 unique classes.")


class LoadClassesTests(unittest.TestCase):
    def test_zero_is_a_class_when_raster_has_no_nodata_value(self):
        data_views = load_data_views(FakeDataset())

        _arr, labels, _resolution, mask = data_views._load_classes("fixture.tif")

        self.assertEqual(labels, {0: "Class 0", 1: "Class 1"})
        np.testing.assert_array_equal(mask, [[False, False]])

    def test_zero_is_excluded_when_declared_as_nodata(self):
        data_views = load_data_views(FakeDataset(nodata=0))

        _arr, labels, _resolution, mask = data_views._load_classes("fixture.tif")

        self.assertEqual(labels, {1: "Class 1"})
        np.testing.assert_array_equal(mask, [[True, False]])


class BuildLandscapeTests(unittest.TestCase):
    def test_missing_nodata_is_not_replaced_with_zero(self):
        landscape = Mock()
        raster_io = load_raster_io(FakeDataset(), landscape)

        raster_io._build_landscape("fixture.tif")

        landscape.assert_called_once_with("fixture.tif", nodata=None, res=(1, 1))

    def test_declared_zero_nodata_is_preserved(self):
        landscape = Mock()
        raster_io = load_raster_io(FakeDataset(nodata=0), landscape)

        raster_io._build_landscape("fixture.tif")

        landscape.assert_called_once_with("fixture.tif", nodata=0, res=(1, 1))


class PreviewRasterTests(unittest.TestCase):
    def test_zero_is_labeled_when_raster_has_no_nodata_value(self):
        raster_io = load_raster_io(FakeDataset())

        figure = raster_io.preview_raster("fixture.tif")

        self.assertEqual(figure.axes.legend_labels, ["Class 0", "Class 1"])

    def test_zero_is_not_labeled_when_declared_as_nodata(self):
        raster_io = load_raster_io(FakeDataset(nodata=0))

        figure = raster_io.preview_raster("fixture.tif")

        self.assertEqual(figure.axes.legend_labels, ["Class 1"])


if __name__ == "__main__":
    unittest.main()
