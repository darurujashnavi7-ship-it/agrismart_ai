import unittest
from io import BytesIO
from unittest.mock import patch

import numpy as np
from PIL import Image

from backend import disease_detection


class FakeModel:
    def predict(self, images, verbose=0):
        return np.array([[0.01, 0.02, 0.97]], dtype=np.float32)


class DiseaseDetectionTests(unittest.TestCase):
    def test_selected_crop_returns_one_confident_diagnosis(self):
        classes=[
            "Apple___Apple_scab",
            "Apple___healthy",
            "Strawberry___Leaf_scorch",
        ]
        image=BytesIO()
        Image.new("RGB",(32,32),(80,140,60)).save(image,format="PNG")
        image.seek(0)

        class AppleModel:
            def predict(self, images, verbose=0):
                return np.array([[0.80, 0.15, 0.05]], dtype=np.float32)

        with patch.object(disease_detection,"load_model",return_value=(AppleModel(),classes)):
            result=disease_detection.predict(image,expected_crop="Apple")

        self.assertTrue(result["reliable"])
        self.assertEqual(result["label"],"Apple___Apple_scab")
        self.assertNotIn("top_predictions",result)

    def test_selected_crop_blocks_confident_prediction_for_another_crop(self):
        classes=[
            "Apple___Apple_scab",
            "Apple___healthy",
            "Strawberry___Leaf_scorch",
        ]
        image=BytesIO()
        Image.new("RGB",(32,32),(80,140,60)).save(image,format="PNG")
        image.seek(0)

        with patch.object(disease_detection,"load_model",return_value=(FakeModel(),classes)):
            result=disease_detection.predict(image,expected_crop="Apple")

        self.assertFalse(result["reliable"])
        self.assertTrue(result["raw_label"].startswith("Apple___"))
        self.assertAlmostEqual(result["confidence"],0.02)

if __name__=="__main__":
    unittest.main()
