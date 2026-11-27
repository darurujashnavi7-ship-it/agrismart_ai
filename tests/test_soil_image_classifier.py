import io
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import backend.soil_image_classifier as classifier


def _image_bytes(size=(224,224),image_format="JPEG"):
    output=io.BytesIO()
    Image.new("RGB",size,color=(120,90,60)).save(output,format=image_format)
    return output.getvalue()


class SoilImageClassifierTests(unittest.TestCase):
    def setUp(self):
        self.previous=(classifier._MODEL,classifier._CLASS_NAMES,classifier._METADATA)
        classifier._MODEL=None
        classifier._CLASS_NAMES=None
        classifier._METADATA=None

    def tearDown(self):
        classifier._MODEL,classifier._CLASS_NAMES,classifier._METADATA=self.previous

    def test_prepare_validates_extension_and_image_dimensions(self):
        image=classifier.prepare_soil_image("sample.jpg",_image_bytes())
        self.assertEqual(image.size,(224,224))

        with self.assertRaisesRegex(ValueError,"JPG, JPEG, or PNG"):
            classifier.prepare_soil_image("sample.gif",_image_bytes())
        with self.assertRaisesRegex(ValueError,"at least 128"):
            classifier.prepare_soil_image("small.png",_image_bytes((64,64),"PNG"))
        with self.assertRaisesRegex(ValueError,"extension does not match"):
            classifier.prepare_soil_image("mismatch.jpg",_image_bytes(image_format="PNG"))
        with self.assertRaisesRegex(ValueError,"10 MB or smaller"):
            classifier.prepare_soil_image("large.jpg",b"x"*(classifier.MAX_IMAGE_BYTES+1))

    def test_model_missing_does_not_create_a_prediction(self):
        with patch.object(classifier,"model_status",return_value=False):
            with self.assertRaisesRegex(FileNotFoundError,"trained soil classifier is missing"):
                classifier.predict_soil_image("sample.jpg",_image_bytes())

    def test_prediction_returns_top_three_and_marks_uncertain_results(self):
        class FakeModel:
            def __init__(self,probabilities):
                self.probabilities=probabilities

            def predict(self,images,verbose=0):
                self.asserted_shape=images.shape
                return np.asarray([self.probabilities])

        class_names=["Alluvial Soil","Arid Soil","Black Soil","Red Soil"]
        classifier._CLASS_NAMES=class_names
        classifier._METADATA={
            "image_size":[224,224],
            "uncertainty_threshold":.55,
            "uncertainty_margin":.1,
        }
        classifier._MODEL=FakeModel([.72,.16,.08,.04])
        result=classifier.predict_soil_image("sample.jpg",_image_bytes())
        self.assertEqual(result["predicted_soil"],"Alluvial Soil")
        self.assertEqual(len(result["top_predictions"]),3)
        self.assertFalse(result["uncertain"])

        classifier._MODEL=FakeModel([.40,.34,.16,.10])
        uncertain=classifier.predict_soil_image("sample.jpg",_image_bytes())
        self.assertIsNone(uncertain["predicted_soil"])
        self.assertTrue(uncertain["uncertain"])


if __name__=="__main__":
    unittest.main()
