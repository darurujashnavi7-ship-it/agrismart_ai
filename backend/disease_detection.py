from pathlib import Path
import json, numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
MODEL_PATH=ROOT/"models"/"disease_model.keras"
LABEL_PATH=ROOT/"models"/"class_names.json"
_model=None; _classes=None

def model_status(): return MODEL_PATH.exists() and LABEL_PATH.exists()

def get_supported_crops():
    if not LABEL_PATH.exists(): return []
    classes=json.loads(LABEL_PATH.read_text(encoding="utf-8"))
    return sorted({label.split("___",1)[0].replace("_"," ") for label in classes if "___" in label})

def load_model():
    global _model,_classes
    if _model is None:
        if not model_status(): raise FileNotFoundError("Disease model is not trained. Run the Colab notebook and copy its exported files into models/.")
        import tensorflow as tf
        _model=tf.keras.models.load_model(MODEL_PATH)
        _classes=json.loads(LABEL_PATH.read_text(encoding="utf-8"))
    return _model,_classes

def predict(image,threshold=0.55,expected_crop=None):
    model,classes=load_model()
    img=Image.open(image).convert("RGB").resize((224,224))
    arr=np.asarray(img,dtype=np.float32)/255.0
    probs=model.predict(np.expand_dims(arr,0),verbose=0)[0]
    if len(probs)!=len(classes):
        raise ValueError("The model output does not match class_names.json. Retrain or restore both model files together.")

    available_indices=list(range(len(classes)))
    if expected_crop:
        expected_prefix=expected_crop.replace(" ","_")+"___"
        available_indices=[index for index,label in enumerate(classes) if label.startswith(expected_prefix)]
        if not available_indices:
            raise ValueError(f"The model has no disease classes for {expected_crop}.")
    idx=max(available_indices,key=lambda index:float(probs[index]))
    conf=float(probs[idx]); label=classes[idx]
    if conf<threshold:
        return {"label":"Uncertain – retake image","confidence":conf,"raw_label":label,"reliable":False}
    crop,disease=label.split("___",1) if "___" in label else ("Unknown",label)
    return {
        "label":label,
        "crop":crop.replace("_"," "),
        "disease":disease.replace("_"," "),
        "confidence":conf,
        "reliable":True,
    }
