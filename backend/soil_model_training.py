import argparse
import hashlib
import json
import random
import subprocess
import tempfile
from datetime import datetime,timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED,ZipFile

from PIL import Image,UnidentifiedImageError


DATASET_URL="https://github.com/Phantom-fs/Soil-Classification-Dataset.git"
DATASET_COMMIT="c3108e9ea8277a2762b0aad587e90cc9722e63d4"
DATASET_DIRECTORY="Orignal-Dataset"
EXPECTED_CLASS_FOLDERS={
    "Alluvial_Soil",
    "Arid_Soil",
    "Black_Soil",
    "Laterite_Soil",
    "Mountain_Soil",
    "Red_Soil",
    "Yellow_Soil",
}
IMAGE_EXTENSIONS={".jpg",".jpeg",".png"}
IMAGE_SIZE=(224,224)
SEED=42


def _download_dataset(destination):
    repository=destination/"dataset"
    subprocess.run(
        [
            "git","clone","--depth","1","--filter=blob:none","--sparse",
            DATASET_URL,str(repository),
        ],
        check=True,
    )
    subprocess.run(
        ["git","-C",str(repository),"fetch","--depth","1","origin",DATASET_COMMIT],
        check=True,
    )
    subprocess.run(
        ["git","-C",str(repository),"checkout","--detach",DATASET_COMMIT],
        check=True,
    )
    subprocess.run(
        ["git","-C",str(repository),"sparse-checkout","set",DATASET_DIRECTORY],
        check=True,
    )
    return repository/DATASET_DIRECTORY


def _resolve_dataset_path(dataset_path,temporary_directory):
    if dataset_path is None:
        return _download_dataset(temporary_directory)
    root=Path(dataset_path).expanduser().resolve()
    if (root/DATASET_DIRECTORY).is_dir():
        return root/DATASET_DIRECTORY
    if root.is_dir():
        return root
    raise FileNotFoundError(f"Soil image dataset directory not found: {root}")


def _valid_image(path):
    try:
        with Image.open(path) as image:
            if image.format not in {"JPEG","PNG"} or image.width<128 or image.height<128:
                return False
            image.verify()
        return True
    except (OSError,UnidentifiedImageError):
        return False


def scan_dataset(dataset_root):
    class_directories={
        path.name:path
        for path in dataset_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    }
    if set(class_directories)!=EXPECTED_CLASS_FOLDERS:
        raise ValueError(
            "Dataset class folders changed. Expected the seven documented original labels: "
            +", ".join(sorted(EXPECTED_CLASS_FOLDERS))
            +"; found: "+", ".join(sorted(class_directories))
        )
    samples=[]
    image_hashes={}
    skipped=0
    for label_id,class_name in enumerate(sorted(EXPECTED_CLASS_FOLDERS)):
        class_root=class_directories[class_name]
        for path in sorted(class_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if not _valid_image(path):
                skipped+=1
                continue
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            previous_label=image_hashes.get(digest)
            if previous_label is not None:
                if previous_label!=class_name:
                    raise ValueError(
                        f"Identical image content occurs under conflicting labels: {path.name}"
                    )
                skipped+=1
                continue
            image_hashes[digest]=class_name
            samples.append({
                "path":str(path),
                "class_folder":class_name,
                "class_name":class_name.replace("_"," "),
                "label_id":label_id,
            })
    counts={class_name:0 for class_name in sorted(EXPECTED_CLASS_FOLDERS)}
    for sample in samples:
        counts[sample["class_folder"]]+=1
    if not samples or any(count<3 for count in counts.values()):
        raise ValueError("Each documented soil class needs at least three valid unique images.")
    return samples,counts,skipped


def split_dataset(samples):
    import numpy as np
    from sklearn.model_selection import train_test_split

    labels=np.asarray([sample["label_id"] for sample in samples])
    train_samples,held_out=train_test_split(
        samples,
        test_size=.30,
        random_state=SEED,
        stratify=labels,
    )
    held_out_labels=np.asarray([sample["label_id"] for sample in held_out])
    validation_samples,test_samples=train_test_split(
        held_out,
        test_size=.50,
        random_state=SEED,
        stratify=held_out_labels,
    )
    return train_samples,validation_samples,test_samples


def _make_dataset(samples,batch_size,training,tf):
    paths=[sample["path"] for sample in samples]
    labels=[sample["label_id"] for sample in samples]
    dataset=tf.data.Dataset.from_tensor_slices((paths,labels))

    def load_image(path,label):
        image_bytes=tf.io.read_file(path)
        image=tf.io.decode_image(image_bytes,channels=3,expand_animations=False)
        image.set_shape([None,None,3])
        image=tf.image.resize(image,IMAGE_SIZE,method="bilinear")
        image=tf.cast(image,tf.float32)
        return image,label

    dataset=dataset.map(load_image,num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        dataset=dataset.shuffle(
            buffer_size=len(samples),
            seed=SEED,
            reshuffle_each_iteration=True,
        )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _build_model(class_count,tf):
    keras=tf.keras
    augmentation=keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(.08),
        keras.layers.RandomZoom(.10),
    ],name="training_augmentation")
    base=keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE,3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable=False
    inputs=keras.Input(shape=(*IMAGE_SIZE,3),name="soil_rgb_0_255")
    x=augmentation(inputs)
    x=keras.layers.Rescaling(1/127.5,offset=-1,name="mobilenetv2_normalization")(x)
    x=base(x,training=False)
    x=keras.layers.GlobalAveragePooling2D()(x)
    x=keras.layers.Dropout(.25)(x)
    outputs=keras.layers.Dense(class_count,activation="softmax",name="soil_class")(x)
    return keras.Model(inputs,outputs),base


def _callbacks(checkpoint_path,tf):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=.3,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def train(args):
    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )
    from sklearn.metrics import ConfusionMatrixDisplay

    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    tf.keras.mixed_precision.set_global_policy("float32")
    output_directory=Path(args.output_dir).expanduser().resolve()
    output_directory.mkdir(parents=True,exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="agrismart-soil-") as temporary_name:
        temporary_directory=Path(temporary_name)
        dataset_root=_resolve_dataset_path(args.dataset_dir,temporary_directory)
        samples,class_counts,skipped=scan_dataset(dataset_root)
        train_samples,validation_samples,test_samples=split_dataset(samples)
        class_names=[
            name.replace("_"," ")
            for name in sorted(EXPECTED_CLASS_FOLDERS)
        ]
        train_counts={
            name:sum(sample["class_name"]==name for sample in train_samples)
            for name in class_names
        }
        validation_counts={
            name:sum(sample["class_name"]==name for sample in validation_samples)
            for name in class_names
        }
        test_counts={
            name:sum(sample["class_name"]==name for sample in test_samples)
            for name in class_names
        }
        train_label_counts=np.bincount(
            [sample["label_id"] for sample in train_samples],
            minlength=len(class_names),
        )
        class_weights={
            label_id:len(train_samples)/(len(class_names)*int(count))
            for label_id,count in enumerate(train_label_counts)
            if count
        }
        print("Unique image counts by class:",class_counts)
        print("Skipped corrupt or exact duplicate files:",skipped)
        print("Split sizes:",len(train_samples),len(validation_samples),len(test_samples))
        print("Training class weights:",class_weights)

        train_ds=_make_dataset(train_samples,args.batch_size,True,tf)
        validation_ds=_make_dataset(validation_samples,args.batch_size,False,tf)
        test_ds=_make_dataset(test_samples,args.batch_size,False,tf)
        model,base=_build_model(len(class_names),tf)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        head_checkpoint=output_directory/"soil_classifier_head_best.keras"
        head_history=model.fit(
            train_ds,
            validation_data=validation_ds,
            epochs=args.epochs,
            callbacks=_callbacks(head_checkpoint,tf),
            class_weight=class_weights,
        )

        base.trainable=True
        for layer in base.layers[:-30]:
            layer.trainable=False
        for layer in base.layers:
            if isinstance(layer,tf.keras.layers.BatchNormalization):
                layer.trainable=False
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        fine_tune_checkpoint=output_directory/"soil_classifier_finetune_best.keras"
        fine_tune_history=model.fit(
            train_ds,
            validation_data=validation_ds,
            epochs=args.fine_tune_epochs,
            callbacks=_callbacks(fine_tune_checkpoint,tf),
            class_weight=class_weights,
        )
        best_head_accuracy=max(head_history.history["val_accuracy"])
        best_fine_tune_accuracy=max(fine_tune_history.history["val_accuracy"])
        selected_checkpoint=(
            fine_tune_checkpoint
            if best_fine_tune_accuracy>=best_head_accuracy
            else head_checkpoint
        )
        model=tf.keras.models.load_model(selected_checkpoint,compile=False)
        probabilities=model.predict(test_ds,verbose=0)
        y_true=np.asarray([sample["label_id"] for sample in test_samples])
        y_pred=np.argmax(probabilities,axis=1)
        report=classification_report(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        matrix=confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
        )
        test_metrics={
            "accuracy":float(accuracy_score(y_true,y_pred)),
            "macro_f1":float(f1_score(y_true,y_pred,average="macro",zero_division=0)),
            "classification_report":report,
            "confusion_matrix":matrix.tolist(),
        }
        print(f"Held-out test accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Held-out macro F1: {test_metrics['macro_f1']:.4f}")

        history_accuracy=(
            head_history.history.get("accuracy",[])
            +fine_tune_history.history.get("accuracy",[])
        )
        history_val_accuracy=(
            head_history.history.get("val_accuracy",[])
            +fine_tune_history.history.get("val_accuracy",[])
        )
        history_loss=(
            head_history.history.get("loss",[])
            +fine_tune_history.history.get("loss",[])
        )
        history_val_loss=(
            head_history.history.get("val_loss",[])
            +fine_tune_history.history.get("val_loss",[])
        )
        figure,axes=plt.subplots(1,2,figsize=(12,4))
        axes[0].plot(history_accuracy,label="Train")
        axes[0].plot(history_val_accuracy,label="Validation")
        axes[0].set_title("Accuracy by epoch")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[1].plot(history_loss,label="Train")
        axes[1].plot(history_val_loss,label="Validation")
        axes[1].set_title("Loss by epoch")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        figure.tight_layout()
        figure.savefig(output_directory/"training_curves.png",dpi=160)
        plt.close(figure)

        figure,axis=plt.subplots(figsize=(10,8))
        ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=class_names,
        ).plot(ax=axis,cmap="Greens",xticks_rotation=45,colorbar=False)
        figure.tight_layout()
        figure.savefig(output_directory/"confusion_matrix.png",dpi=160)
        plt.close(figure)

        model_path=output_directory/"soil_classifier.keras"
        model.save(model_path)
        metadata={
            "model_version":"mobilenetv2-soil-7class-v1",
            "architecture":"MobileNetV2 transfer learning with fine-tuning",
            "image_size":list(IMAGE_SIZE),
            "dataset_url":"https://github.com/Phantom-fs/Soil-Classification-Dataset",
            "dataset_commit":DATASET_COMMIT,
            "dataset_paper_doi":"10.1016/j.engappai.2025.111425",
            "dataset_license":"CC BY-NC 4.0; non-commercial use only",
            "class_names":class_names,
            "source_image_counts":class_counts,
            "training_counts":train_counts,
            "validation_counts":validation_counts,
            "test_counts":test_counts,
            "skipped_corrupt_or_duplicate_images":skipped,
            "split_seed":SEED,
            "split_percentages":{"train":70,"validation":15,"test":15},
            "training_completed_at_utc":datetime.now(timezone.utc).isoformat(),
            "uncertainty_threshold":.55,
            "uncertainty_margin":.10,
            "uncertainty_note":(
                "Conservative starting thresholds, not calibrated for out-of-distribution or non-soil images. "
                "Validate with independent field and non-soil photos before deployment."
            ),
            "test_metrics":test_metrics,
        }
        (output_directory/"soil_class_names.json").write_text(
            json.dumps(class_names,indent=2),
            encoding="utf-8",
        )
        (output_directory/"soil_classifier_metadata.json").write_text(
            json.dumps(metadata,indent=2),
            encoding="utf-8",
        )
        with ZipFile(output_directory/"AgriSmart_Soil_Classifier.zip","w",ZIP_DEFLATED) as bundle:
            for file_name in (
                "soil_classifier.keras",
                "soil_class_names.json",
                "soil_classifier_metadata.json",
                "training_curves.png",
                "confusion_matrix.png",
            ):
                bundle.write(output_directory/file_name,arcname=file_name)
        print("Export bundle:",output_directory/"AgriSmart_Soil_Classifier.zip")
        print("Classification report:")
        for class_name in class_names:
            item=report[class_name]
            print(
                f"{class_name}: precision={item['precision']:.3f}, "
                f"recall={item['recall']:.3f}, f1={item['f1-score']:.3f}, "
                f"support={int(item['support'])}"
            )


def main():
    parser=argparse.ArgumentParser(
        description="Train and evaluate the AgriSmart seven-class soil image model."
    )
    parser.add_argument("--dataset-dir",help="Dataset root or its Orignal-Dataset directory.")
    parser.add_argument("--output-dir",default="soil_classifier_export")
    parser.add_argument("--epochs",type=int,default=12)
    parser.add_argument("--fine-tune-epochs",type=int,default=8)
    parser.add_argument("--batch-size",type=int,default=32)
    args=parser.parse_args()
    if args.epochs<1 or args.fine_tune_epochs<1 or args.batch_size<1:
        parser.error("Epoch counts and batch size must be positive.")
    train(args)


if __name__=="__main__":
    main()
