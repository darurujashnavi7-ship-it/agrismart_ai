import io
import json
import math
import threading
from pathlib import Path

import numpy as np
from PIL import Image,ImageOps,UnidentifiedImageError


ROOT=Path(__file__).resolve().parents[1]
MODEL_PATH=ROOT/"models"/"soil_classifier.keras"
CLASS_NAMES_PATH=ROOT/"models"/"soil_class_names.json"
METADATA_PATH=ROOT/"models"/"soil_classifier_metadata.json"
MAX_IMAGE_BYTES=10*1024*1024
MAX_IMAGE_PIXELS=25_000_000
MIN_IMAGE_DIMENSION=128
SUPPORTED_SUFFIXES={".jpg",".jpeg",".png"}
EXPECTED_CLASS_NAMES=(
    "Alluvial Soil",
    "Arid Soil",
    "Black Soil",
    "Laterite Soil",
    "Mountain Soil",
    "Red Soil",
    "Yellow Soil",
)
_MODEL=None
_CLASS_NAMES=None
_METADATA=None
_MODEL_LOCK=threading.Lock()


def model_status():
    return all(path.is_file() for path in (MODEL_PATH,CLASS_NAMES_PATH,METADATA_PATH))


def prepare_soil_image(file_name,file_bytes):
    suffix=Path(file_name).suffix.lower() if isinstance(file_name,str) else ""
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError("Upload a JPG, JPEG, or PNG soil photo.")
    if not isinstance(file_bytes,bytes) or not file_bytes:
        raise ValueError("The uploaded image is empty.")
    if len(file_bytes)>MAX_IMAGE_BYTES:
        raise ValueError("Soil photos must be 10 MB or smaller.")
    try:
        with Image.open(io.BytesIO(file_bytes)) as source:
            expected_format="PNG" if suffix==".png" else "JPEG"
            if source.format!=expected_format:
                raise ValueError("The file extension does not match the image contents.")
            width,height=source.size
            if width<MIN_IMAGE_DIMENSION or height<MIN_IMAGE_DIMENSION:
                raise ValueError(
                    f"Image dimensions must be at least {MIN_IMAGE_DIMENSION} x {MIN_IMAGE_DIMENSION} pixels."
                )
            if width*height>MAX_IMAGE_PIXELS:
                raise ValueError("The image dimensions are too large to analyze safely.")
            source.verify()
        with Image.open(io.BytesIO(file_bytes)) as source:
            image=ImageOps.exif_transpose(source).convert("RGB")
    except ValueError:
        raise
    except (UnidentifiedImageError,OSError) as error:
        raise ValueError("The image could not be decoded. Try a clear JPG or PNG photograph.") from error
    return image


def _load_model():
    global _MODEL,_CLASS_NAMES,_METADATA
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                if not model_status():
                    raise FileNotFoundError(
                        "The trained soil classifier is missing. Run "
                        "notebooks/Soil_Image_Classification_Colab.ipynb and extract its "
                        "model files into models/."
                    )
                try:
                    class_names=json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
                    metadata=json.loads(METADATA_PATH.read_text(encoding="utf-8"))
                except (OSError,json.JSONDecodeError) as error:
                    raise ValueError("Soil model metadata is missing or invalid; export the model bundle again.") from error
                if (
                    not isinstance(class_names,list)
                    or not class_names
                    or any(not isinstance(name,str) or not name.strip() for name in class_names)
                    or len(set(class_names))!=len(class_names)
                ):
                    raise ValueError("soil_class_names.json must contain unique class-name strings.")
                if class_names!=list(EXPECTED_CLASS_NAMES):
                    raise ValueError(
                        "The soil model must use the seven documented dataset labels in their "
                        "training order. Unsupported image classes must not be added by guessing."
                    )
                if not isinstance(metadata,dict):
                    raise ValueError("soil_classifier_metadata.json must contain a JSON object.")
                if metadata.get("class_names")!=class_names:
                    raise ValueError(
                        "soil_classifier_metadata.json and soil_class_names.json do not agree; "
                        "install all files from one model export."
                    )
                try:
                    import tensorflow as tf

                    model=tf.keras.models.load_model(MODEL_PATH,compile=False)
                except (ImportError,OSError,ValueError) as error:
                    raise RuntimeError(f"Could not load the soil classifier: {error}") from error
                output_shape=model.output_shape
                if isinstance(output_shape,list) or output_shape[-1]!=len(class_names):
                    raise ValueError(
                        "The soil model output does not match soil_class_names.json. "
                        "Install a matching model export."
                    )
                _MODEL=model
                _CLASS_NAMES=class_names
                _METADATA=metadata
    return _MODEL,_CLASS_NAMES,_METADATA


def predict_soil_image(file_name,file_bytes):
    image=prepare_soil_image(file_name,file_bytes)
    model,class_names,metadata=_load_model()
    image_size=metadata.get("image_size",[224,224])
    if (
        not isinstance(image_size,list)
        or len(image_size)!=2
        or any(not isinstance(size,int) or size<1 for size in image_size)
    ):
        raise ValueError("The model export has an invalid image_size setting.")
    resized=image.resize((image_size[1],image_size[0]),Image.Resampling.BILINEAR)
    pixels=np.asarray(resized,dtype=np.float32)
    probabilities=np.asarray(
        model.predict(np.expand_dims(pixels,axis=0),verbose=0)[0],
        dtype=np.float64,
    )
    if (
        probabilities.shape!=(len(class_names),)
        or not np.isfinite(probabilities).all()
        or np.any(probabilities<0)
        or np.any(probabilities>1)
    ):
        raise ValueError("The model returned invalid class probabilities.")
    total=float(probabilities.sum())
    if not math.isfinite(total) or total<=0:
        raise ValueError("The model returned an empty probability distribution.")
    probabilities/=total
    top_indices=np.argsort(probabilities)[::-1][:min(3,len(class_names))]
    top_predictions=[
        {"soil_type":class_names[int(index)],"confidence":float(probabilities[index])}
        for index in top_indices
    ]
    top_confidence=top_predictions[0]["confidence"]
    margin=(
        top_confidence-top_predictions[1]["confidence"]
        if len(top_predictions)>1 else top_confidence
    )
    try:
        confidence_threshold=float(metadata.get("uncertainty_threshold",0.55))
        margin_threshold=float(metadata.get("uncertainty_margin",0.10))
    except (TypeError,ValueError) as error:
        raise ValueError("The soil model export has invalid uncertainty thresholds.") from error
    if (
        not math.isfinite(confidence_threshold)
        or not 0<=confidence_threshold<=1
        or not math.isfinite(margin_threshold)
        or not 0<=margin_threshold<=1
    ):
        raise ValueError("The soil model export has invalid uncertainty thresholds.")
    uncertain=top_confidence<confidence_threshold or margin<margin_threshold
    return {
        "predicted_soil":None if uncertain else top_predictions[0]["soil_type"],
        "confidence":top_confidence,
        "margin":margin,
        "uncertain":uncertain,
        "top_predictions":top_predictions,
        "model_version":metadata.get("model_version","Unspecified"),
        "test_metrics":metadata.get("test_metrics",{}),
        "image_width":image.width,
        "image_height":image.height,
    }
