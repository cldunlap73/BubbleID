import unittest

import numpy as np
import torch

from BubbleID.indexing import filter_segmentation_outputs


class FilterSegmentationOutputsTest(unittest.TestCase):
    def test_filters_outputs_with_explicit_numpy_and_torch_indices(self):
        masks = torch.tensor(
            [
                [[True, False], [False, False]],
                [[False, True], [False, False]],
                [[False, False], [True, True]],
            ]
        )
        scores = torch.tensor([0.9, 0.2, 0.8])
        class_values = [0, 1, 0]

        kept_masks, kept_classes, base_masks = filter_segmentation_outputs(
            masks, scores, class_values, 0.5
        )

        self.assertTrue(torch.equal(kept_masks, masks[[0, 2]]))
        np.testing.assert_array_equal(kept_classes, np.array([0, 0]))
        self.assertTrue(torch.equal(base_masks, masks[[0, 2]]))


if __name__ == "__main__":
    unittest.main()
