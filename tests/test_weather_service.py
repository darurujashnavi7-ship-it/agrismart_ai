import unittest
from unittest.mock import patch

import backend.weather_service as weather_service
from backend.weather_service import farming_advice,weather_background_theme,weather_condition


class WeatherServiceTests(unittest.TestCase):
    def test_weather_condition_describes_rain(self):
        icon,description=weather_condition(63)

        self.assertTrue(icon)
        self.assertEqual(description,"Moderate rain")

    def test_weather_background_theme_uses_weather_codes_and_priority(self):
        cases={
            0:"sunny",
            2:"cloudy",
            45:"foggy",
            61:"rainy",
            95:"stormy",
        }

        for code,expected in cases.items():
            with self.subTest(code=code):
                self.assertEqual(
                    weather_background_theme({"current":{"weather_code":code}}),
                    expected,
                )

        self.assertEqual(
            weather_background_theme({
                "current":{"weather_code":95,"precipitation":1,"wind_speed_10m":60}
            }),
            "stormy",
        )
        self.assertEqual(
            weather_background_theme({
                "current":{"weather_code":45,"precipitation":1,"wind_speed_10m":60}
            }),
            "rainy",
        )
        self.assertEqual(
            weather_background_theme({
                "current":{"weather_code":45,"wind_speed_10m":60}
            }),
            "foggy",
        )
        self.assertEqual(
            weather_background_theme({
                "current":{"weather_code":2,"wind_speed_10m":31}
            }),
            "windy",
        )

    def test_weather_background_theme_detects_rain_and_configurable_wind(self):
        self.assertEqual(
            weather_background_theme({"current":{"weather_code":0,"precipitation":0.2}}),
            "rainy",
        )
        self.assertEqual(
            weather_background_theme(
                {"current":{"weather_code":0,"wind_speed_10m":31}},
                wind_threshold_kmh=30,
            ),
            "windy",
        )
        self.assertEqual(
            weather_background_theme(
                {"current":{"weather_code":0,"wind_speed_10m":30}},
                wind_threshold_kmh=30,
            ),
            "sunny",
        )
        with patch.object(weather_service,"WINDY_THRESHOLD_KMH",20):
            self.assertEqual(
                weather_background_theme(
                    {"current":{"weather_code":0,"wind_speed_10m":21}}
                ),
                "windy",
            )

    def test_farming_advice_reports_multiple_risk_factors(self):
        data={
            "current":{
                "temperature_2m":36,
                "relative_humidity_2m":85,
                "wind_speed_10m":18,
            },
            "daily":{
                "precipitation_probability_max":[90],
                "precipitation_sum":[20],
                "uv_index_max":[9],
            },
        }

        advice=farming_advice(data)

        self.assertEqual(len(advice),5)
        self.assertTrue(any("Rain is likely" in item for item in advice))
        self.assertTrue(any("Wind is strong" in item for item in advice))
        self.assertTrue(any("Humidity is high" in item for item in advice))
        self.assertTrue(any("High temperatures" in item for item in advice))
        self.assertTrue(any("UV levels are very high" in item for item in advice))

    def test_farming_advice_has_general_guidance_in_mild_weather(self):
        advice=farming_advice({"current":{},"daily":{}})

        self.assertEqual(len(advice),1)
        self.assertIn("soil moisture",advice[0])


if __name__=="__main__":
    unittest.main()
