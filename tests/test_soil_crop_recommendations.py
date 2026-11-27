import unittest

from backend.soil_crop_recommendations import (
    load_crop_profiles,
    recommend_crops,
    resolve_season,
)


def _profile():
    return {
        "crop":"Test crop",
        "soil_classes":["Black Soil"],
        "regions":["Test district","Test state"],
        "seasons":["Kharif"],
        "temperature_c_range":[20,30],
        "seasonal_rainfall_mm_range":[300,700],
        "irrigation_options":["Supplemental"],
        "water_requirement":"Moderate, based on the test-only profile.",
        "why":"Test-only sourced-profile explanation.",
        "growing_conditions":"Test conditions.",
        "improvements":["Use local soil test results."],
        "cultivation":"Follow local extension guidance.",
        "sources":[{"title":"Test profile","url":"https://example.test/profile"}],
    }


class SoilCropRecommendationTests(unittest.TestCase):
    def test_empty_config_does_not_fabricate_crop_rankings(self):
        config=load_crop_profiles()
        result=recommend_crops(
            "Black Soil","Test district, Test state","Kharif","Supplemental",
            {
                "seasonal_mean_temperature_c":25,
                "seasonal_rainfall_mm":500,
                "source":"Test local climate bulletin",
            },
            config["profiles"],
        )
        self.assertEqual(result["recommendations"],[])
        self.assertIn("no sourced",result["reason"])

    def test_rule_based_score_uses_configured_soil_region_season_climate_and_water(self):
        result=recommend_crops(
            "Black Soil","Test district, Test state","Kharif","Supplemental",
            {
                "seasonal_mean_temperature_c":25,
                "seasonal_rainfall_mm":500,
                "source":"Test local climate bulletin",
            },
            [_profile()],
        )
        self.assertEqual(len(result["recommendations"]),1)
        self.assertEqual(result["recommendations"][0]["crop"],"Test crop")
        self.assertEqual(result["recommendations"][0]["score"],100)
        self.assertIn("not yield predictions",result["reason"])

    def test_unmatched_and_missing_climate_profiles_are_not_recommended(self):
        profile=_profile()
        profile["soil_classes"]=["Red Soil"]
        result=recommend_crops(
            "Black Soil","Test district, Test state","Kharif","Supplemental",
            {"seasonal_mean_temperature_c":25,"seasonal_rainfall_mm":500},
            [profile],
        )
        self.assertEqual(result["recommendations"],[])

        missing_climate=recommend_crops(
            "Black Soil","Test district, Test state","Kharif","Supplemental",
            None,[_profile()],
        )
        self.assertEqual(missing_climate["recommendations"],[])
        self.assertIn("seasonal",missing_climate["reason"].lower())

    def test_india_season_default_is_explicitly_month_based(self):
        self.assertEqual(resolve_season("Auto (India calendar; verify locally)",7),"Kharif")
        self.assertEqual(resolve_season("Auto (India calendar; verify locally)",1),"Rabi")
        self.assertEqual(resolve_season("Auto (India calendar; verify locally)",4),"Zaid")
        self.assertEqual(resolve_season("Rabi",7),"Rabi")


if __name__=="__main__":
    unittest.main()
