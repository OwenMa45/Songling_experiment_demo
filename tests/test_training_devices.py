import unittest
from piper_titration.training import validate_device_layout


class TrainingDeviceTests(unittest.TestCase):
    def test_single_data_parallel_and_fsdp(self):
        for layout in ((1,1,1),(4,1,4),(4,4,4),(2,2,2)):
            validate_device_layout(*layout)

    def test_invalid_layouts_fail_before_weight_restore(self):
        for layout in ((1,4,4),(4,3,4),(1,2,1),(0,1,1),(4,0,4),(True,1,1)):
            with self.subTest(layout=layout), self.assertRaises(ValueError):
                validate_device_layout(*layout)
