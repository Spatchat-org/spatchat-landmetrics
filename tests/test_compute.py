import sys
import types
import unittest
from unittest.mock import patch


def unexpected_raster_call(_file):
    raise AssertionError("Raster processing should not run in this routing test")


raster_io = types.ModuleType("raster_io")
raster_io._build_landscape = unexpected_raster_call
raster_io.answer_metadata = unexpected_raster_call
raster_io.count_classes = unexpected_raster_call

with patch.dict(sys.modules, {"raster_io": raster_io}):
    import compute


class RunComputeMetricsTests(unittest.TestCase):
    def test_class_level_request_preserves_cross_level_metrics(self):
        with patch.object(
            compute, "compute_class_only_text", return_value="result"
        ) as compute_class:
            result = compute.run_compute_metrics(
                "fixture.tif", ["pland", "edge_density"], "class"
            )

        self.assertEqual(result, "result")
        compute_class.assert_called_once_with(
            "fixture.tif", ["pland", "edge_density"]
        )


if __name__ == "__main__":
    unittest.main()
