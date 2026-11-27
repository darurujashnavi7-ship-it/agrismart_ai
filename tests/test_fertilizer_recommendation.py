import unittest

from backend.fertilizer_recommendation import get_fertilizer_names


class FertilizerRecommendationTests(unittest.TestCase):
    def test_all_model_crops_have_fertilizer_names(self):
        labels=[
            "Apple___Apple_scab",
            "Blueberry___healthy",
            "Cherry_(including_sour)___Powdery_mildew",
            "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
            "Grape___Black_rot",
            "Orange___Haunglongbing_(Citrus_greening)",
            "Peach___Bacterial_spot",
            "Pepper,_bell___Bacterial_spot",
            "Potato___Early_blight",
            "Raspberry___healthy",
            "Soybean___healthy",
            "Squash___Powdery_mildew",
            "Strawberry___Leaf_scorch",
            "Tomato___Late_blight",
        ]

        for label in labels:
            with self.subTest(label=label):
                names=get_fertilizer_names(label)
                self.assertTrue(names)
                self.assertTrue(all(isinstance(name,str) and name for name in names))

    def test_disease_class_does_not_change_crop_fertilizer_names(self):
        self.assertEqual(
            get_fertilizer_names("Tomato___healthy"),
            get_fertilizer_names("Tomato___Late_blight"),
        )

    def test_unknown_crop_is_reported(self):
        with self.assertRaises(ValueError):
            get_fertilizer_names("Unknown___healthy")


if __name__=="__main__":
    unittest.main()
