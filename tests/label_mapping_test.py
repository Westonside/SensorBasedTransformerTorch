import unittest
from preprocess.dataset_loading import remap_classes
import numpy as np
class TestLabelMapping(unittest.TestCase):


    def test_remap_shl_one(self):
        align_all_classes = ['Walk', 'Upstair', 'Downstair', 'Sit', 'Stand', 'Lay', 'Jump', 'Run', 'Bike', 'Car', 'Bus',
                             'Train', 'Subway']
        SHL = [4, 0, 7, 8, 9, 10, 11, 12]
        train_labels = np.array([*SHL, *SHL, *SHL])
        test_labels = np.array(SHL)
        remapped_labels_train,remapped_labels_test, remapped_labels = remap_classes(train_labels, test_labels, align_all_classes)
        self.assertEqual(len(SHL), len(np.unique(np.hstack((remapped_labels_train,remapped_labels_test)))))

if __name__ == '__main__':
    unittest.main()