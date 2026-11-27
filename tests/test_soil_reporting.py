import io
import unittest

from pypdf import PdfReader
from reportlab.pdfgen import canvas

from backend.soil_reporting import create_soil_report_pdf,extract_soil_values


class SoilReportingTests(unittest.TestCase):
    def test_csv_report_extracts_supported_labeled_measurements(self):
        content=(
            b"Nitrogen (kg/ha),P (kg/ha),K (kg/ha),Soil pH,Soil moisture (%),"
            b"Soil temperature (C),EC (dS/m),Organic carbon (%)\n"
            b"120,18,240,6.8,32,22,0.5,0.7\n"
        )

        values=extract_soil_values("sample.csv",content)

        self.assertEqual(values["nitrogen"],120)
        self.assertEqual(values["phosphorus"],18)
        self.assertEqual(values["ec"],0.5)
        self.assertEqual(values["organic_carbon"],0.7)

    def test_csv_with_multiple_samples_is_rejected(self):
        content=b"Nitrogen (kg/ha)\n100\n120\n"

        with self.assertRaisesRegex(ValueError,"one soil-sample row"):
            extract_soil_values("samples.csv",content)

    def test_oxide_basis_values_are_not_silently_treated_as_elemental_pk(self):
        with self.assertRaisesRegex(ValueError,"not automatically converted"):
            extract_soil_values(
                "sample.csv",
                b"N (kg/ha),P2O5 (kg/ha),K2O (kg/ha)\n100,20,30\n",
            )

        output=io.BytesIO()
        document=canvas.Canvas(output)
        document.drawString(40,760,"P2O5 (kg/ha): 18")
        document.save()
        with self.assertRaisesRegex(ValueError,"not automatically converted"):
            extract_soil_values("sample.pdf",output.getvalue())

    def test_pdf_values_with_unsupported_units_are_not_interpreted(self):
        output=io.BytesIO()
        document=canvas.Canvas(output)
        document.drawString(40,760,"Available Nitrogen (mg/kg): 120")
        document.save()

        with self.assertRaisesRegex(ValueError,"unsupported unit"):
            extract_soil_values("sample.pdf",output.getvalue())

    def test_pdf_report_extracts_labeled_values_and_rejects_scans(self):
        output=io.BytesIO()
        document=canvas.Canvas(output)
        document.drawString(40,760,"Available Nitrogen (kg/ha): 120")
        document.drawString(40,740,"P: 18")
        document.drawString(40,720,"K: 240")
        document.drawString(40,700,"Soil pH: 6.8")
        document.drawString(40,680,"Organic Carbon (%): 0.7")
        document.save()

        values=extract_soil_values("sample.pdf",output.getvalue())

        self.assertEqual(values["nitrogen"],120)
        self.assertEqual(values["ph"],6.8)
        self.assertEqual(values["organic_carbon"],0.7)
        blank_pdf=io.BytesIO()
        canvas.Canvas(blank_pdf).save()
        with self.assertRaisesRegex(ValueError,"No supported measurements"):
            extract_soil_values("scanned.pdf",blank_pdf.getvalue())

    def test_pdf_export_contains_values_rules_and_disclaimer(self):
        analysis={
            "values":{
                "nitrogen":120,
                "phosphorus":18,
                "potassium":240,
                "ph":6.8,
                "moisture":32,
                "temperature":22,
                "ec":0.5,
                "organic_carbon":0.7,
            },
            "nutrient_status":{"nitrogen":"Medium","phosphorus":"Low","potassium":"Medium"},
            "ph_status":"Neutral",
            "moisture_status":"Not configured",
            "organic_carbon_status":"Medium",
            "crop":"Wheat",
            "soil_type":"Loamy soil",
            "location":"Test district",
            "reference_name":"Test reference",
            "reference_sources":[{"title":"Test reference","url":"https://example.test"}],
            "health_score":50,
            "health_status":"Monitor and improve",
            "recommendations":[],
            "crop_suitability":[],
        }

        report=create_soil_report_pdf(analysis,"Farmer-entered values")
        text="\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(report)).pages)

        self.assertTrue(report.startswith(b"%PDF"))
        self.assertIn("120 kg/ha",text)
        self.assertIn("not a laboratory test",text)
        self.assertIn("Test reference",text)


if __name__=="__main__":
    unittest.main()
