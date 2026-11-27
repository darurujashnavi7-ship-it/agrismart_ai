import unittest

from backend.soil_analysis import (
    analyze_soil,
    load_soil_config,
    select_soil_reference,
    soil_assistant_context,
    soil_assistant_fallback,
    validate_soil_inputs,
)


REFERENCE={
    "name":"Test-only configurable ranges",
    "nutrients":{
        "nitrogen":{"low_upper":10,"medium_upper":30},
        "phosphorus":{"low_upper":5,"medium_upper":15},
        "potassium":{"low_upper":20,"medium_upper":50},
    },
    "ph":{"acidic_upper":6,"alkaline_lower":8},
    "organic_carbon":{"low_upper":0.3,"medium_upper":0.8},
    "moisture":{"dry_upper":20,"wet_lower":70},
}
CROP_PROFILES=[
    {
        "name":f"Crop {index}",
        "ph_min":6,
        "ph_max":8,
        "soil_types":["Loamy soil"],
    }
    for index in range(6)
]
SOIL_VALUES={
    "nitrogen":20,
    "phosphorus":10,
    "potassium":35,
    "ph":7,
    "moisture":45,
    "temperature":22,
    "ec":0.5,
    "organic_carbon":0.5,
}


class SoilAnalysisTests(unittest.TestCase):
    def test_analysis_uses_supplied_ranges_and_marks_rule_based_screen(self):
        result=analyze_soil(
            SOIL_VALUES,"Loamy soil","Crop 0","Test district",REFERENCE,CROP_PROFILES
        )

        self.assertEqual(result["nutrient_status"],{
            "nitrogen":"Medium",
            "phosphorus":"Medium",
            "potassium":"Medium",
        })
        self.assertEqual(result["ph_status"],"Neutral")
        self.assertEqual(result["moisture_status"],"Adequate")
        self.assertEqual(result["health_score"],100)
        self.assertEqual(len(result["crop_suitability"]),5)
        self.assertEqual(result["reference_name"],"Test-only configurable ranges")

    def test_low_nutrient_recommends_options_without_application_rates(self):
        values={**SOIL_VALUES,"nitrogen":5}
        result=analyze_soil(
            values,"Loamy soil","Crop 0","Test district",REFERENCE,CROP_PROFILES
        )

        self.assertIn("Nitrogen",result["deficiencies"])
        nitrogen_recommendation=next(
            item for item in result["recommendations"]
            if item["title"].startswith("Nitrogen source")
        )
        self.assertIn("configured reference",nitrogen_recommendation["reason"])
        self.assertNotRegex(nitrogen_recommendation["reason"],r"\d+\s*(?:kg|g|ton)")

    def test_invalid_measurement_and_invalid_reference_are_rejected(self):
        with self.assertRaisesRegex(ValueError,"between 0 and 14"):
            validate_soil_inputs(
                {**SOIL_VALUES,"ph":float("nan")},
                "Loamy soil",
                "Crop 0",
                "Test district",
            )
        invalid_reference={
            **REFERENCE,
            "nutrients":{
                **REFERENCE["nutrients"],
                "nitrogen":{"low_upper":30,"medium_upper":10},
            },
        }
        with self.assertRaisesRegex(ValueError,"Reference thresholds are invalid"):
            analyze_soil(
                SOIL_VALUES,"Loamy soil","Crop 0","Test district",
                invalid_reference,CROP_PROFILES,
            )

    def test_tnau_profile_is_location_scoped_and_includes_documented_cutoffs(self):
        config=load_soil_config()
        tamil_nadu=select_soil_reference(
            config,"Loamy soil","Rice","Salem, Tamil Nadu"
        )
        other_state=select_soil_reference(
            config,"Loamy soil","Rice","Bengaluru, Karnataka"
        )

        self.assertIn("TNAU",tamil_nadu["name"])
        self.assertEqual(tamil_nadu["nutrients"]["nitrogen"]["low_upper"],240)
        self.assertIsNone(tamil_nadu["organic_carbon"]["low_upper"])
        self.assertIn("Local laboratory",other_state["name"])
        self.assertIsNone(other_state["nutrients"]["nitrogen"]["low_upper"])

    def test_tnau_available_npk_boundaries_match_the_published_medium_band(self):
        config=load_soil_config()
        reference=select_soil_reference(
            config,"Loamy soil","Rice","Salem, Tamil Nadu"
        )
        values={
            **SOIL_VALUES,
            "nitrogen":480,
            "phosphorus":22,
            "potassium":280,
        }

        result=analyze_soil(
            values,"Loamy soil","Rice","Salem, Tamil Nadu",
            reference,[],
        )
        self.assertEqual(set(result["nutrient_status"].values()),{"Medium"})

        above_nitrogen_boundary=analyze_soil(
            {**values,"nitrogen":480.1},
            "Loamy soil","Rice","Salem, Tamil Nadu",
            reference,[],
        )
        self.assertEqual(above_nitrogen_boundary["nutrient_status"]["nitrogen"],"High")

    def test_unconfigured_values_are_not_classified_or_scored(self):
        reference=select_soil_reference(
            load_soil_config(),"Loamy soil","Wheat","Bengaluru, Karnataka"
        )
        result=analyze_soil(
            SOIL_VALUES,"Loamy soil","Wheat","Bengaluru, Karnataka",
            reference,[],
        )

        self.assertEqual(set(result["nutrient_status"].values()),{"Not configured"})
        self.assertEqual(result["ph_status"],"Not configured")
        self.assertEqual(result["organic_carbon_status"],"Not configured")
        self.assertIsNone(result["health_score"])
        self.assertEqual(result["health_status"],"Not configured")
        self.assertEqual(result["crop_suitability"],[])

    def test_medium_class_includes_published_upper_boundary(self):
        result=analyze_soil(
            {**SOIL_VALUES,"nitrogen":30},
            "Loamy soil","Crop 0","Test district",REFERENCE,CROP_PROFILES,
        )

        self.assertEqual(result["nutrient_status"]["nitrogen"],"Medium")

    def test_offline_assistant_uses_supplied_analysis_without_claiming_lab_test(self):
        result=analyze_soil(
            {**SOIL_VALUES,"nitrogen":5},
            "Loamy soil",
            "Crop 0",
            "Test district",
            REFERENCE,
            CROP_PROFILES,
        )

        self.assertIn("Nitrogen: low",soil_assistant_fallback("Explain nitrogen",result))
        context=soil_assistant_context(result)
        self.assertIn("not independently verified laboratory results",context)
        self.assertIn("Do not invent missing measurements",context)


if __name__=="__main__":
    unittest.main()
