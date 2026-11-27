import html
import io
from datetime import datetime

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle,getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as PdfImage,
    Flowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from backend.soil_image_classifier import prepare_soil_image


def _safe(value):
    text=str(value).encode("ascii","replace").decode("ascii")
    return html.escape(text)


def _bullet_list(items,styles):
    if not items:
        return [Paragraph("No profile-specific advice is configured.",styles["BodyText"])]
    return [
        Paragraph("- "+_safe(item),styles["BodyText"])
        for item in items
    ]


class _AgriSmartLogo(Flowable):
    def __init__(self):
        super().__init__()
        self.width=.55*inch
        self.height=.48*inch

    def draw(self):
        drawing=self.canv
        drawing.setFillColor(colors.HexColor("#14532d"))
        drawing.roundRect(0,0,self.width,self.height,7,stroke=0,fill=1)
        drawing.setStrokeColor(colors.white)
        drawing.setLineWidth(1.7)
        drawing.line(self.width*.5,self.height*.17,self.width*.5,self.height*.68)
        left_leaf=drawing.beginPath()
        left_leaf.moveTo(self.width*.5,self.height*.48)
        left_leaf.curveTo(self.width*.22,self.height*.45,self.width*.22,self.height*.78,self.width*.46,self.height*.79)
        left_leaf.curveTo(self.width*.55,self.height*.72,self.width*.56,self.height*.59,self.width*.5,self.height*.48)
        left_leaf.close()
        right_leaf=drawing.beginPath()
        right_leaf.moveTo(self.width*.5,self.height*.58)
        right_leaf.curveTo(self.width*.78,self.height*.55,self.width*.78,self.height*.88,self.width*.54,self.height*.89)
        right_leaf.curveTo(self.width*.45,self.height*.81,self.width*.44,self.height*.69,self.width*.5,self.height*.58)
        right_leaf.close()
        drawing.drawPath(left_leaf,stroke=1,fill=0)
        drawing.drawPath(right_leaf,stroke=1,fill=0)


def create_soil_image_report_pdf(
    file_name,
    image_bytes,
    prediction,
    soil_guidance,
    crop_result,
    farmer_name="",
    location="",
    season="",
    irrigation="",
    weather_summary=None,
    climate_source="",
):
    image=prepare_soil_image(file_name,image_bytes)
    photo=io.BytesIO()
    image.thumbnail((900,700))
    image.save(photo,format="JPEG",quality=88)
    photo.seek(0)

    output=io.BytesIO()
    document=SimpleDocTemplate(
        output,
        pagesize=letter,
        rightMargin=.62*inch,
        leftMargin=.62*inch,
        topMargin=.55*inch,
        bottomMargin=.55*inch,
    )
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="AgriSmartSoilTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#14532d"),
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="AgriSmartSection",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#166534"),
        spaceBefore=12,
        spaceAfter=6,
    ))
    story=[
        Table(
            [[
                _AgriSmartLogo(),
                Paragraph("AgriSmart Precision Agriculture",styles["AgriSmartSoilTitle"]),
            ]],
            colWidths=[.7*inch,6*inch],
            style=TableStyle([("VALIGN",(0,0),(-1,-1),"MIDDLE")]),
        ),
        Paragraph("Image-Based Soil Classification Report",styles["Heading2"]),
        Paragraph(
            "Generated: "+_safe(datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")),
            styles["BodyText"],
        ),
    ]
    if farmer_name.strip():
        story.append(Paragraph("Farmer: "+_safe(farmer_name.strip()),styles["BodyText"]))
    if location.strip():
        story.append(Paragraph("Farm location: "+_safe(location.strip()),styles["BodyText"]))
    story.extend([
        Spacer(1,8),
        PdfImage(photo,width=4.8*inch,height=3.45*inch,kind="proportional"),
        Spacer(1,10),
        Paragraph("Image classification",styles["AgriSmartSection"]),
    ])
    if prediction.get("uncertain"):
        story.append(Paragraph(
            "Result: Uncertain. The classifier did not meet its configured confidence and "
            "top-class separation thresholds. No soil type is asserted.",
            styles["BodyText"],
        ))
    else:
        story.append(Paragraph(
            "Predicted dataset label: <b>"+_safe(prediction["predicted_soil"])+"</b>",
            styles["BodyText"],
        ))
    story.append(Paragraph(
        f"Top-class model confidence: {prediction['confidence']*100:.1f}%. "
        "This is not a calibrated probability that the sample is soil.",
        styles["BodyText"],
    ))
    model_version=prediction.get("model_version")
    if model_version:
        story.append(Paragraph("Model version: "+_safe(model_version),styles["BodyText"]))
    test_accuracy=prediction.get("test_metrics",{}).get("accuracy")
    if isinstance(test_accuracy,(int,float)):
        story.append(Paragraph(
            f"Held-out test-set accuracy recorded by training: {test_accuracy*100:.1f}%. "
            "This metric is dataset-specific and is not expected field-photo accuracy.",
            styles["BodyText"],
        ))
    table_data=[["Rank","Dataset class","Model score"]]
    for index,item in enumerate(prediction.get("top_predictions",[]),start=1):
        table_data.append([
            str(index),
            _safe(item["soil_type"]),
            f"{item['confidence']*100:.1f}%",
        ])
    if len(table_data)>1:
        table=Table(table_data,colWidths=[.65*inch,4.55*inch,1.3*inch],repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#14532d")),
            ("TEXTCOLOR",(0,0),(-1,0),colors.white),
            ("GRID",(0,0),(-1,-1),.4,colors.HexColor("#cbd5cc")),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f1f8f2")]),
            ("VALIGN",(0,0),(-1,-1),"TOP"),
            ("PADDING",(0,0),(-1,-1),6),
        ]))
        story.extend([Spacer(1,7),table])

    crop_guidance=soil_guidance.get("crop_guidance",{})
    story.append(Paragraph("Crops to consider for this soil class",styles["AgriSmartSection"]))
    if prediction.get("uncertain"):
        story.append(Paragraph(
            "Crop examples are withheld because the soil image prediction is uncertain.",
            styles["BodyText"],
        ))
    else:
        story.append(Paragraph(
            "These are general source-backed examples, not site-specific recommendations or guarantees.",
            styles["BodyText"],
        ))
        if crop_guidance.get("crops"):
            story.extend(_bullet_list(crop_guidance["crops"],styles))
        if crop_guidance.get("note"):
            story.append(Paragraph(_safe(crop_guidance["note"]),styles["BodyText"]))
        if crop_guidance.get("source_url"):
            story.append(Paragraph(
                "Source: "+_safe(crop_guidance.get("source_title",""))+
                " - "+_safe(crop_guidance["source_url"]),
                styles["BodyText"],
            ))

    recommendations=crop_result.get("recommendations",[])
    if recommendations:
        story.append(Paragraph("Location-specific crop screening",styles["AgriSmartSection"]))
        story.append(Paragraph(
            "Rule-based suitability scores from configured regional profiles. "
            "They are not probabilities, yield predictions, or guarantees.",
            styles["BodyText"],
        ))
        story.append(Paragraph(
            "Season: "+_safe(season or "Not specified")+"; irrigation: "+
            _safe(irrigation or "Not specified"),
            styles["BodyText"],
        ))
        if climate_source:
            story.append(Paragraph(
                "Seasonal climate input source: "+_safe(climate_source),
                styles["BodyText"],
            ))
        climate=weather_summary or {}
        if climate:
            story.append(Paragraph(
                "Weather context (short-range forecast, not seasonal climate normals): "+
                _safe(climate.get("summary","Unavailable")),
                styles["BodyText"],
            ))
        for index,crop in enumerate(recommendations,start=1):
            story.append(Paragraph(
                f"<b>{index}. {_safe(crop['crop'])} - {crop['score']}/100</b>",
                styles["BodyText"],
            ))
            story.append(Paragraph(_safe(crop["why"]),styles["BodyText"]))
            story.append(Paragraph(
                "Season/conditions: "+_safe(crop["growing_conditions"])+
                ". Water: "+_safe(crop["water_requirement"])+
                ". Improvements: "+_safe("; ".join(crop["improvements"])),
                styles["BodyText"],
            ))
            story.append(Paragraph("Cultivation: "+_safe(crop["cultivation"]),styles["BodyText"]))
            for source in crop.get("sources",[]):
                story.append(Paragraph(
                    "Source: "+_safe(source["title"])+" - "+_safe(source["url"]),
                    styles["BodyText"],
                ))
            story.append(Spacer(1,5))

    story.extend([
        Paragraph("Important limitations",styles["AgriSmartSection"]),
        Paragraph(
            "This closed-set image model compares the photograph only with its trained dataset "
            "classes. It can misclassify non-soil images and field photos unlike its training data. "
            "A photograph cannot measure texture, NPK, pH, organic carbon, electrical conductivity, "
            "moisture, salinity, or fertilizer requirements. Obtain a representative laboratory soil "
            "test before making amendment or crop decisions. Do not infer fertilizer quantities from "
            "this image report.",
            styles["BodyText"],
        ),
        Paragraph(
            "Image dataset attribution: Sheth et al. (2025), "
            "https://doi.org/10.1016/j.engappai.2025.111425. "
            "Dataset license: CC BY-NC 4.0; educational/non-commercial use only.",
            styles["BodyText"],
        ),
    ])
    document.build(story)
    return output.getvalue()
