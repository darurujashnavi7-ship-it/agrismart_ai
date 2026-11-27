import html
import io
import re

import pandas as pd
from pypdf import PdfReader
from pypdf.errors import FileNotDecryptedError,PdfReadError
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


FIELD_ALIASES={
    "nitrogen":{
        "n","nitrogen","availablen","availablenitrogen","nitrogenkgha","nkgha",
    },
    "phosphorus":{
        "p","pkgha","phosphorus","availablep","availablephosphorus",
        "phosphoruskgha",
    },
    "potassium":{
        "k","kkgha","potassium","availablek","availablepotassium",
        "potassiumkgha",
    },
    "ph":{"ph","soilph","soilphvalue"},
    "moisture":{"moisture","soilmoisture","soilmoisturepercent"},
    "temperature":{"soiltemperature","soiltemperaturec","temperature"},
    "ec":{"ec","ecdsm","electricalconductivity","electricalconductivitydsm"},
    "organic_carbon":{"organiccarbon","organiccarbonpercent","oc"},
}

PDF_LABELS={
    "nitrogen":r"(?:available\s+nitrogen|nitrogen|n)",
    "phosphorus":r"(?:available\s+phosphorus|phosphorus|p)",
    "potassium":r"(?:available\s+potassium|potassium|k)",
    "ph":r"(?:soil\s+ph|ph)",
    "moisture":r"(?:soil\s+moisture|moisture)",
    "temperature":r"(?:soil\s+temperature|temperature)",
    "ec":r"(?:electrical\s+conductivity|ec)",
    "organic_carbon":r"(?:organic\s+carbon|oc)",
}


def _normalized_label(value):
    return re.sub(r"[^a-z0-9]+","",str(value).lower())


def _escaped(value):
    normalized=str(value).encode("ascii","replace").decode("ascii")
    return html.escape(normalized)


def _number(value):
    match=re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)",str(value))
    if not match:
        return None
    try:
        number=float(match.group())
    except ValueError:
        return None
    return number if number==number and abs(number)!=float("inf") else None


def _values_from_csv(file_bytes):
    try:
        frame=pd.read_csv(io.BytesIO(file_bytes))
    except (pd.errors.EmptyDataError,pd.errors.ParserError,UnicodeDecodeError) as error:
        raise ValueError("The CSV is empty or is not readable as text.") from error

    column_map={}
    for column in frame.columns:
        normalized=_normalized_label(column)
        if "p2o5" in normalized or "k2o" in normalized:
            raise ValueError(
                "P2O5/K2O values are not automatically converted to P/K. "
                "Confirm the nutrient basis with the soil report and enter compatible values manually."
            )
        for field,aliases in FIELD_ALIASES.items():
            if normalized in aliases:
                column_map[field]=column
                break
    if column_map:
        if len(frame)!=1:
            raise ValueError("Use a CSV with one soil-sample row; multiple sample rows are ambiguous.")
        values={}
        for field,column in column_map.items():
            number=_number(frame.iloc[0][column])
            if number is not None:
                values[field]=number
        return values

    try:
        rows=pd.read_csv(io.BytesIO(file_bytes),header=None)
    except (pd.errors.EmptyDataError,pd.errors.ParserError,UnicodeDecodeError) as error:
        raise ValueError("The CSV is empty or is not readable as text.") from error
    values={}
    for _,row in rows.iterrows():
        if len(row)<2:
            continue
        normalized=_normalized_label(row.iloc[0])
        if "p2o5" in normalized or "k2o" in normalized:
            raise ValueError(
                "P2O5/K2O values are not automatically converted to P/K. "
                "Confirm the nutrient basis with the soil report and enter compatible values manually."
            )
        for field,aliases in FIELD_ALIASES.items():
            if normalized in aliases:
                number=_number(row.iloc[1])
                if number is not None:
                    values[field]=number
                break
    return values


def _values_from_pdf(file_bytes):
    try:
        reader=PdfReader(io.BytesIO(file_bytes))
        if len(reader.pages)>20:
            raise ValueError("PDF reports are limited to 20 pages.")
        text="\n".join(page.extract_text() or "" for page in reader.pages)
    except ValueError:
        raise
    except (PdfReadError,FileNotDecryptedError,OSError,TypeError) as error:
        raise ValueError("The PDF could not be read. Try a text-based PDF or enter values manually.") from error

    if re.search(r"\b(?:p2o5|k2o)\b",text,re.IGNORECASE):
        raise ValueError(
            "P2O5/K2O values are not automatically converted to P/K. "
            "Confirm the nutrient basis with the soil report and enter compatible values manually."
        )

    values={}
    supported_units={
        "nitrogen":{"kgha"},
        "phosphorus":{"kgha"},
        "potassium":{"kgha"},
        "ph":{"ph"},
        "moisture":{"percent"},
        "temperature":{"c","celsius"},
        "ec":{"dsm"},
        "organic_carbon":{"percent"},
    }
    for field,label in PDF_LABELS.items():
        pattern=re.compile(
            rf"(?im)^\s*{label}\b(?:\s*\((?P<unit>[^)]*)\))?\s*[:=\-]?\s*"
            rf"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+))"
        )
        match=pattern.search(text)
        if match:
            unit_text=match.group("unit")
            normalized_unit=_normalized_label(unit_text) if unit_text else ""
            if normalized_unit and normalized_unit not in supported_units[field]:
                raise ValueError(
                    f"The PDF reports {field.replace('_',' ')} in an unsupported unit. "
                    "No unit conversion is performed; enter the compatible value manually."
                )
            values[field]=float(match.group("value"))
    return values


def extract_soil_values(file_name,file_bytes):
    if len(file_bytes)>10*1024*1024:
        raise ValueError("Soil reports must be 10 MB or smaller.")
    suffix=file_name.rsplit(".",1)[-1].lower() if "." in file_name else ""
    if suffix=="csv":
        values=_values_from_csv(file_bytes)
    elif suffix=="pdf":
        values=_values_from_pdf(file_bytes)
    else:
        raise ValueError("Upload a CSV or PDF soil report.")
    if not values:
        raise ValueError(
            "No supported measurements were recognized. Enter the values manually; "
            "automatic extraction supports labeled soil fields only."
        )
    return values


def create_soil_report_pdf(analysis,source_label):
    output=io.BytesIO()
    score_label=(
        f"{analysis['health_score']}/100 ({_escaped(analysis['health_status'])})"
        if analysis["health_score"] is not None
        else f"Not configured ({_escaped(analysis['health_status'])})"
    )
    document=SimpleDocTemplate(
        output,
        pagesize=letter,
        rightMargin=.65*inch,
        leftMargin=.65*inch,
        topMargin=.65*inch,
        bottomMargin=.65*inch,
    )
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="SoilTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#14532d"),
        spaceAfter=12,
    ))
    story=[
        Paragraph("AgriSmart Soil Analysis Summary",styles["SoilTitle"]),
        Paragraph(
            "This report summarizes values entered or extracted from the farmer's uploaded file. "
            "It is not a laboratory test, certified diagnosis, or fertilizer prescription.",
            styles["BodyText"],
        ),
        Spacer(1,12),
    ]
    values=analysis["values"]
    table_data=[
        ["Field","Reported value","Configured interpretation"],
        ["Nitrogen",f"{values['nitrogen']:g} kg/ha",analysis["nutrient_status"]["nitrogen"]],
        ["Phosphorus",f"{values['phosphorus']:g} kg/ha",analysis["nutrient_status"]["phosphorus"]],
        ["Potassium",f"{values['potassium']:g} kg/ha",analysis["nutrient_status"]["potassium"]],
        ["Soil pH",f"{values['ph']:g}",analysis["ph_status"]],
        ["Moisture",f"{values['moisture']:g}%",analysis["moisture_status"]],
        ["Soil temperature",f"{values['temperature']:g} °C","Recorded; no crop-stage rating applied"],
        ["Electrical conductivity",f"{values['ec']:g} dS/m","Recorded; interpret using the lab method"],
        ["Organic carbon",f"{values['organic_carbon']:g}%",analysis["organic_carbon_status"]],
    ]
    table=Table(table_data,colWidths=[1.8*inch,1.65*inch,3.2*inch],repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#14532d")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("GRID",(0,0),(-1,-1),.4,colors.HexColor("#cbd5cc")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f1f8f2")]),
        ("VALIGN",(0,0),(-1,-1),"TOP"),
        ("PADDING",(0,0),(-1,-1),7),
    ]))
    story.extend([
        table,
        Spacer(1,12),
        Paragraph(f"Crop: {_escaped(analysis['crop'])}",styles["BodyText"]),
        Paragraph(f"Soil type: {_escaped(analysis['soil_type'])}",styles["BodyText"]),
        Paragraph(f"Farm location: {_escaped(analysis['location'])}",styles["BodyText"]),
        Paragraph(f"Input source: {_escaped(source_label)}",styles["BodyText"]),
        Paragraph(f"Reference: {_escaped(analysis['reference_name'])}",styles["BodyText"]),
        Paragraph(
            f"Rule-based soil indicator: {score_label}. "
            "This is not a standardized soil-health score.",
            styles["BodyText"],
        ),
        Spacer(1,10),
        Paragraph("Recommendations (no application rates are inferred)",styles["Heading2"]),
    ])
    for source in analysis.get("reference_sources",[]):
        story.append(Paragraph(
            "Reference source: "+_escaped(source["title"])+" - "+_escaped(source["url"]),
            styles["BodyText"],
        ))
        story.append(Spacer(1,4))
    for recommendation in analysis["recommendations"]:
        story.append(Paragraph(
            f"<b>{_escaped(recommendation['title'])}</b>: "
            f"{_escaped(recommendation['reason'])}",
            styles["BodyText"],
        ))
        story.append(Spacer(1,5))
    story.extend([
        Spacer(1,8),
        Paragraph("Rule-based crop suitability (soil-only screen; not an ML prediction)",styles["Heading2"]),
    ])
    for item in analysis["crop_suitability"]:
        story.append(Paragraph(
            f"<b>{_escaped(item['crop'])} - {item['score']}/100</b>: "
            f"{_escaped(' '.join(item['reasons']))} "
            f"Improvements: {_escaped(' '.join(item['improvements']))}",
            styles["BodyText"],
        ))
        story.append(Spacer(1,5))
    document.build(story)
    return output.getvalue()
