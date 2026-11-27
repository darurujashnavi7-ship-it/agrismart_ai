FERTILIZERS_BY_CROP={
    "Apple":["Urea","Single Superphosphate (SSP)","Muriate of Potash (MOP)"],
    "Blueberry":["Ammonium Sulfate","Sulphate of Potash (SOP)"],
    "Cherry (including sour)":["Urea","Single Superphosphate (SSP)","Muriate of Potash (MOP)"],
    "Corn (maize)":["Urea","Diammonium Phosphate (DAP)","Muriate of Potash (MOP)"],
    "Grape":["Urea","Diammonium Phosphate (DAP)","Muriate of Potash (MOP)"],
    "Orange":["Urea","Diammonium Phosphate (DAP)","Muriate of Potash (MOP)"],
    "Peach":["Urea","Single Superphosphate (SSP)","Muriate of Potash (MOP)"],
    "Pepper, bell":["NPK 19:19:19","Calcium Nitrate","Muriate of Potash (MOP)"],
    "Potato":["NPK 10:26:26","Urea","Muriate of Potash (MOP)"],
    "Raspberry":["Urea","Single Superphosphate (SSP)","Muriate of Potash (MOP)"],
    "Soybean":["Diammonium Phosphate (DAP)","Single Superphosphate (SSP)","Muriate of Potash (MOP)"],
    "Squash":["NPK 19:19:19","Urea","Muriate of Potash (MOP)"],
    "Strawberry":["NPK 19:19:19","Calcium Nitrate","Sulphate of Potash (SOP)"],
    "Tomato":["NPK 19:19:19","Calcium Nitrate","Muriate of Potash (MOP)"],
}

def get_fertilizer_names(label):
    crop=label.split("___",1)[0].replace("_"," ")
    if crop not in FERTILIZERS_BY_CROP:
        raise ValueError(f"No fertilizer list is configured for crop '{crop}'.")
    return FERTILIZERS_BY_CROP[crop]
