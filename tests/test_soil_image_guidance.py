import unittest

from backend.soil_image_guidance import (
    load_soil_guidance,
    soil_image_assistant_fallback,
    soil_image_question_is_relevant,
)


class SoilImageAssistantTests(unittest.TestCase):
    def test_fallback_explains_classification_without_claiming_lab_results(self):
        prediction={
            "predicted_soil":"Black Soil",
            "confidence":0.82,
            "uncertain":False,
        }
        response=soil_image_assistant_fallback(
            "What does this soil image tell me?",
            prediction,
            load_soil_guidance(),
            {"recommendations":[],"reason":"No sourced local crop profiles are configured."},
        )
        self.assertIn("Black Soil",response)
        self.assertIn("Cotton",response)
        self.assertIn("https://ncert.nic.in/textbook/pdf/jess101.pdf",response)
        self.assertIn("cannot measure NPK",response)

    def test_only_source_supported_crop_examples_are_listed_by_soil_class(self):
        config=load_soil_guidance()
        self.assertEqual(
            config["classes"]["Black Soil"]["crop_guidance"]["crops"],
            ["Cotton"],
        )
        self.assertEqual(
            config["classes"]["Mountain Soil"]["crop_guidance"]["crops"],
            [],
        )
        self.assertIn(
            "vary with elevation",
            config["classes"]["Mountain Soil"]["crop_guidance"]["note"],
        )

    def test_uncertain_prediction_is_not_reported_as_a_definite_class(self):
        response=soil_image_assistant_fallback(
            "Explain this soil classification",
            {
                "predicted_soil":None,
                "confidence":0.41,
                "uncertain":True,
                "top_predictions":[
                    {"soil_type":"Red Soil","confidence":0.41},
                    {"soil_type":"Yellow Soil","confidence":0.30},
                ],
            },
            load_soil_guidance(),
            {"recommendations":[],"reason":"No matching crop profiles."},
        )
        self.assertIn("uncertain",response)
        self.assertIn("Tentative model candidates",response)
        self.assertNotIn("The image model predicted",response)
        self.assertNotIn("Cotton",response)

    def test_unrelated_agriculture_faq_is_not_captured_by_soil_fallback(self):
        self.assertFalse(soil_image_question_is_relevant("When should I plant wheat?"))
        self.assertIsNone(
            soil_image_assistant_fallback(
                "When should I plant wheat?",
                {"predicted_soil":"Black Soil","confidence":0.8},
                load_soil_guidance(),
                {"recommendations":[]},
            )
        )


if __name__=="__main__":
    unittest.main()
