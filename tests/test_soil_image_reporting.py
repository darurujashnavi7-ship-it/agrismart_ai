import io
import json
import unittest
from pathlib import Path

from PIL import Image
from pypdf import PdfReader

from backend.soil_image_reporting import create_soil_image_report_pdf


def _image_bytes():
    output=io.BytesIO()
    Image.new("RGB",(256,256),color=(130,100,70)).save(output,format="JPEG")
    return output.getvalue()


class SoilImageReportingTests(unittest.TestCase):
    def test_report_embeds_image_and_disclaims_image_only_measurements(self):
        guidance_path=Path(__file__).resolve().parents[1]/"data"/"soil_image_guidance.json"
        guidance_config=json.loads(guidance_path.read_text(encoding="utf-8"))
        prediction={
            "predicted_soil":"Black Soil",
            "confidence":.73,
            "uncertain":False,
            "top_predictions":[
                {"soil_type":"Black Soil","confidence":.73},
                {"soil_type":"Red Soil","confidence":.17},
                {"soil_type":"Alluvial Soil","confidence":.10},
            ],
            "model_version":"test-only",
            "test_metrics":{"accuracy":.61},
        }
        crop_result={"recommendations":[],"reason":"No sourced profiles configured."}
        report=create_soil_image_report_pdf(
            "soil.jpg",
            _image_bytes(),
            prediction,
            guidance_config["classes"]["Black Soil"],
            crop_result,
            farmer_name="Test farmer",
            location="Test district",
            season="Kharif",
            irrigation="Supplemental",
        )
        reader=PdfReader(io.BytesIO(report))
        text="\n".join(page.extract_text() or "" for page in reader.pages)
        self.assertTrue(report.startswith(b"%PDF"))
        self.assertIn("Black Soil",text)
        self.assertIn("Cotton",text)
        self.assertNotIn("Common characteristics",text)
        self.assertIn("not a calibrated probability",text)
        self.assertIn("cannot measure",text)
        self.assertTrue(any(page.images for page in reader.pages))

    def test_uncertain_prediction_withholds_crop_examples(self):
        guidance_path=Path(__file__).resolve().parents[1]/"data"/"soil_image_guidance.json"
        guidance_config=json.loads(guidance_path.read_text(encoding="utf-8"))
        report=create_soil_image_report_pdf(
            "soil.jpg",
            _image_bytes(),
            {
                "predicted_soil":None,
                "confidence":.41,
                "uncertain":True,
                "top_predictions":[
                    {"soil_type":"Black Soil","confidence":.41},
                    {"soil_type":"Red Soil","confidence":.3},
                    {"soil_type":"Alluvial Soil","confidence":.2},
                ],
            },
            guidance_config["classes"]["Black Soil"],
            {"recommendations":[],"reason":"No sourced profiles configured."},
        )
        text="\n".join(
            page.extract_text() or ""
            for page in PdfReader(io.BytesIO(report)).pages
        )
        self.assertIn("Crop examples are withheld",text)
        self.assertNotIn("Cotton",text)


if __name__=="__main__":
    unittest.main()
