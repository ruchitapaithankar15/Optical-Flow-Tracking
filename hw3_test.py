import hw3
import unittest
import numpy as np
import sys
import os
import imageio

sys.path.insert(0, os.path.dirname(__file__))


class Homework3Test(unittest.TestCase):

    def setUp(self):
        self.I = imageio.imread(
            "test_inputs/test_img.jpg")[:, :, :3].astype(np.float32) / 255.0
        self.H = imageio.imread(
            "test_inputs/test_img_2.jpg")[:, :, :3].astype(np.float32) / 255.0
        self.img = imageio.imread(
            "test_inputs/32.jpg")[:, :, :3].astype(np.float32) / 255.0
        self.imgTranslated = imageio.imread(
            "test_inputs/32_t.4.2.jpg")[:, :, :3].astype(np.float32) / 255.0

    def test_lucas_kanade(self):
        expected_displacement = np.array([0.1136673, 0.3789053])
        expected_AtA = np.array(
            [[1.1758014,  -0.20582686], [-0.20582686,  2.6653478]])
        expected_Atb = np.array([0.05566128, 0.9865186])

        displacement, AtA, Atb = hw3.lucas_kanade(self.H, self.I)

        self.assertAlmostEqual(np.linalg.norm(displacement - expected_displacement), 0.0,
                               places=4, msg='The displacement returned ' + str(displacement) + ' was not as expected ' + str(expected_displacement))
        self.assertAlmostEqual(np.linalg.norm(AtA - expected_AtA), 0.0,
                               places=4, msg='The AtA returned ' + str(AtA) + ' was not as expected ' + str(expected_AtA))
        self.assertAlmostEqual(np.linalg.norm(Atb - expected_Atb), 0.0,
                               places=4, msg='The Atb returned ' + str(Atb) + ' was not as expected ' + str(expected_Atb))

    def test_iterative_lucas_kanade(self):

        expected_displacement = np.array([0.04543689, 0.04000662])

        displacement = hw3.iterative_lucas_kanade(self.H, self.I, 2)

        self.assertAlmostEqual(np.linalg.norm(displacement - expected_displacement), 0.0,
                               places=4, msg='The displacement returned ' + str(displacement) + ' was not as expected ' + str(expected_displacement))

    def test_gaussian_pyramid(self):
        result = hw3.gaussian_pyramid(self.img, 4)
        self.assertEqual(len(result), 4, msg='The number of images returned ' +
                         str(len(result)) + ' was not as expected ' + str(4))
        expectedResolutions = [32, 16, 8, 4]
        for i, img in enumerate(result):
            self.assertEqual(img.shape[1], expectedResolutions[i], msg='Level %d image width (%d) is not as expected (%d)' % (
                i, img.shape[1], expectedResolutions[i]))
            self.assertEqual(img.shape[0], expectedResolutions[i], msg='Level %d image height (%d) is not as expected (%d)' % (
                i, img.shape[0], expectedResolutions[i]))

    def test_pyramid_lucas_kanade(self):

        expected_displacement = np.array([4.179228, 2.0011392])

        displacement = hw3.pyramid_lucas_kanade(
            self.img, self.imgTranslated, np.array([0.0, 0.0]), 3, 20)

        self.assertAlmostEqual(displacement[0], expected_displacement[0],
                               places=2, msg='The displacement returned ' + str(displacement) + ' was not as expected ' + str(expected_displacement))
        self.assertAlmostEqual(displacement[1], expected_displacement[1],
                               places=2, msg='The displacement returned ' + str(displacement) + ' was not as expected ' + str(expected_displacement))


if __name__ == '__main__':
    unittest.main()
