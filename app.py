from __future__ import annotations

import base64
import io
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("LOKY_MAX_CPU_COUNT", os.getenv("NUMBER_OF_PROCESSORS", "1"))


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = Path(r"mnist.pkl")
DEFAULT_DATASET_PATH = Path(r"mnist_train.csv")
MODEL_PATH = Path(os.getenv("MNIST_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
DATASET_PATH = Path(os.getenv("MNIST_DATA_PATH", str(DEFAULT_DATASET_PATH)))
HTML_TEMPLATE_PATH = BASE_DIR / "index.html"
PREPROCESS_BUNDLE_PATH = BASE_DIR / "mnist_inference_bundle.pkl"

IMAGE_SIZE = 28
TARGET_DIGIT_BOX = 20
PIXEL_THRESHOLD = 20
TOP_PREDICTIONS = 3
RESAMPLE_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS


class ArtifactError(RuntimeError):
    """Raised when a required artifact is missing or incompatible."""


def load_model():
    if not MODEL_PATH.exists():
        raise ArtifactError(f"MNIST model file not found: {MODEL_PATH}")

    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


def load_raw_feature_columns(model) -> tuple[list[str], str]:
    if DATASET_PATH.exists():
        header = pd.read_csv(DATASET_PATH, nrows=0)
        feature_columns = [str(column) for column in header.columns[1:]]
        if len(feature_columns) == IMAGE_SIZE * IMAGE_SIZE:
            return feature_columns, "training CSV"

    if hasattr(model, "feature_names_in_"):
        feature_columns = [str(column) for column in model.feature_names_in_]
        if len(feature_columns) == IMAGE_SIZE * IMAGE_SIZE:
            return feature_columns, "model metadata"

    feature_columns = [
        f"{row}x{column}" for row in range(1, IMAGE_SIZE + 1) for column in range(1, IMAGE_SIZE + 1)
    ]
    return feature_columns, "generated fallback"


def describe_model(model) -> tuple[str, str]:
    model_name = f"{type(model).__module__}.{type(model).__name__}"

    if hasattr(model, "steps"):
        step_names = [str(name) for name, _ in model.steps]
        return model_name, " -> ".join(step_names)

    return model_name, type(model).__name__


def load_preprocessing_bundle(n_components: int) -> dict:
    if PREPROCESS_BUNDLE_PATH.exists():
        with PREPROCESS_BUNDLE_PATH.open("rb") as bundle_file:
            bundle = pickle.load(bundle_file)

        if (
            bundle.get("dataset_path") == str(DATASET_PATH)
            and bundle.get("components") == n_components
            and bundle.get("feature_columns")
            and bundle.get("scaler") is not None
            and bundle.get("pca") is not None
        ):
            return bundle

    if not DATASET_PATH.exists():
        raise ArtifactError(
            "The model expects PCA features, but the training CSV was not found. "
            f"Expected it at {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]

    X_train, _, _, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    pca = PCA(n_components=n_components)
    pca.fit(X_train_scaled)

    bundle = {
        "dataset_path": str(DATASET_PATH),
        "components": n_components,
        "feature_columns": X.columns.tolist(),
        "scaler": scaler,
        "pca": pca,
    }

    with PREPROCESS_BUNDLE_PATH.open("wb") as bundle_file:
        pickle.dump(bundle, bundle_file)

    return bundle


def detect_inference_mode(model) -> tuple[str, int | None]:
    expected_features = getattr(model, "n_features_in_", None)
    if expected_features is not None:
        expected_features = int(expected_features)

    if hasattr(model, "steps"):
        return "direct_model", expected_features

    if expected_features in (None, IMAGE_SIZE * IMAGE_SIZE):
        return "direct_model", expected_features

    if 0 < expected_features < IMAGE_SIZE * IMAGE_SIZE:
        return "reconstructed_preprocessing", expected_features

    raise ArtifactError(
        f"Loaded model expects {expected_features} features, but this app only knows how to "
        f"prepare {IMAGE_SIZE * IMAGE_SIZE} raw MNIST pixels or a reduced PCA representation."
    )


def bootstrap_service() -> dict:
    model = load_model()
    feature_columns, feature_source = load_raw_feature_columns(model)

    if len(feature_columns) != IMAGE_SIZE * IMAGE_SIZE:
        raise ArtifactError(
            f"Expected 784 input features, but found {len(feature_columns)}. "
            "Please check that the MNIST CSV header still contains the 28x28 pixel columns."
        )

    inference_mode, expected_features = detect_inference_mode(model)
    model_name, pipeline_summary = describe_model(model)
    service = {
        "model": model,
        "feature_columns": feature_columns,
        "scaler": None,
        "pca": None,
        "inference_mode": inference_mode,
        "meta": {
            "model_path": str(MODEL_PATH),
            "dataset_path": str(DATASET_PATH),
            "feature_source": feature_source,
            "feature_count": len(feature_columns),
            "expected_model_features": expected_features,
            "model_name": model_name,
            "pipeline_summary": pipeline_summary,
            "inference_mode_label": "Direct raw MNIST pixels",
            "inference_mode_description": (
                "The saved artifact accepts the 784 original pixel values, so the app sends "
                "the centered 28 x 28 digit straight into the model."
            ),
        },
    }

    if inference_mode == "reconstructed_preprocessing":
        bundle = load_preprocessing_bundle(expected_features)
        service["feature_columns"] = bundle["feature_columns"]
        service["scaler"] = bundle["scaler"]
        service["pca"] = bundle["pca"]
        service["meta"].update(
            {
                "feature_source": "training CSV + reconstructed notebook preprocessing",
                "inference_mode_label": "Notebook-style StandardScaler + PCA",
                "inference_mode_description": (
                    "The saved artifact expects reduced PCA features, so the app rebuilds the "
                    "notebook preprocessing from the training CSV before prediction."
                ),
                "bundle_path": str(PREPROCESS_BUNDLE_PATH),
            }
        )

    return service


def decode_canvas_image(data_url: str) -> Image.Image:
    if not isinstance(data_url, str) or "," not in data_url:
        raise ValueError("Canvas image is missing or invalid.")

    header, encoded = data_url.split(",", 1)
    if not header.startswith("data:image"):
        raise ValueError("Only image payloads are supported.")

    try:
        image_bytes = base64.b64decode(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError("Unable to decode the canvas image.") from exc

    with Image.open(io.BytesIO(image_bytes)) as source:
        rgba_image = source.convert("RGBA")

    background = Image.new("RGBA", rgba_image.size, (0, 0, 0, 255))
    return Image.alpha_composite(background, rgba_image).convert("L")


def shift_to_center(image_array: np.ndarray) -> np.ndarray:
    coords = np.argwhere(image_array > 0)
    if coords.size == 0:
        raise ValueError("Please draw a digit before predicting.")

    center_y, center_x = coords.mean(axis=0)
    target_center = (IMAGE_SIZE - 1) / 2
    shift_y = int(round(target_center - center_y))
    shift_x = int(round(target_center - center_x))

    shifted = np.roll(image_array, shift=(shift_y, shift_x), axis=(0, 1))

    if shift_y > 0:
        shifted[:shift_y, :] = 0
    elif shift_y < 0:
        shifted[shift_y:, :] = 0

    if shift_x > 0:
        shifted[:, :shift_x] = 0
    elif shift_x < 0:
        shifted[:, shift_x:] = 0

    return shifted


def preprocess_digit_image(data_url: str) -> tuple[np.ndarray, np.ndarray]:
    grayscale = decode_canvas_image(data_url)
    pixel_array = np.asarray(grayscale, dtype=np.uint8)
    mask = pixel_array > PIXEL_THRESHOLD

    if not mask.any():
        raise ValueError("Please draw a digit before predicting.")

    ys, xs = np.where(mask)
    top, bottom = int(ys.min()), int(ys.max()) + 1
    left, right = int(xs.min()), int(xs.max()) + 1
    cropped = grayscale.crop((left, top, right, bottom))

    width, height = cropped.size
    scale = TARGET_DIGIT_BOX / max(width, height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cropped.resize((resized_width, resized_height), RESAMPLE_LANCZOS)

    canvas = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)
    offset_x = (IMAGE_SIZE - resized_width) // 2
    offset_y = (IMAGE_SIZE - resized_height) // 2
    canvas.paste(resized, (offset_x, offset_y))

    centered_image = shift_to_center(np.asarray(canvas, dtype=np.float32))
    flat_pixels = centered_image.reshape(1, -1)
    return flat_pixels, centered_image.astype(np.uint8)


def image_array_to_data_url(image_array: np.ndarray) -> str:
    preview = Image.fromarray(image_array.astype(np.uint8), mode="L")
    buffer = io.BytesIO()
    preview.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def prepare_model_input(service: dict, raw_pixels: np.ndarray):
    raw_frame = pd.DataFrame(raw_pixels, columns=service["feature_columns"])

    if service["inference_mode"] == "direct_model":
        return raw_frame

    scaled = service["scaler"].transform(raw_frame)
    return service["pca"].transform(scaled)


def resolve_model_classes(model, class_count: int):
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "steps") and model.steps:
        classes = getattr(model.steps[-1][1], "classes_", None)

    if classes is None:
        return list(range(class_count))

    return classes


app = Flask(__name__)
SERVICE = None
BOOT_ERROR = None

try:
    SERVICE = bootstrap_service()
except Exception as exc:  # pragma: no cover - startup error is shown in the UI
    BOOT_ERROR = str(exc)


def get_service() -> dict:
    if SERVICE is None:
        raise ArtifactError(BOOT_ERROR or "The MNIST service is not ready yet.")
    return SERVICE


@app.get("/")
def index():
    return render_template(
        "index.html",
        model_ready=SERVICE is not None,
        boot_error=BOOT_ERROR,
        meta=SERVICE["meta"] if SERVICE else None,
    )


@app.get("/assets/<path:filename>")
def asset(filename: str):
    if filename not in {"styles.css", "script.js"}:
        return jsonify({"error": "Asset not found."}), 404

    return send_from_directory(BASE_DIR, filename)


@app.get("/health")
def health():
    if SERVICE is None:
        return jsonify({"status": "error", "message": BOOT_ERROR}), 500

    return jsonify({"status": "ok", "meta": SERVICE["meta"]})


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    try:
        service = get_service()
        image_data = payload.get("image")
        raw_pixels, preview_pixels = preprocess_digit_image(image_data)
        model = service["model"]
        model_input = prepare_model_input(service, raw_pixels)

        predicted_digit = int(model.predict(model_input)[0])

        probabilities = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(model_input)[0]

        top_predictions = []
        confidence = None
        if probabilities is not None:
            ranked = np.argsort(probabilities)[::-1][:TOP_PREDICTIONS]
            classes = resolve_model_classes(model, len(probabilities))
            top_predictions = [
                {
                    "digit": int(classes[index]),
                    "confidence": round(float(probabilities[index] * 100), 2),
                }
                for index in ranked
            ]
            confidence = round(float(probabilities[ranked[0]] * 100), 2)

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except ArtifactError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception:
        return jsonify(
            {"error": "Prediction failed. Please check that `pipe.pkl` and the dataset header are valid."}
        ), 500

    return jsonify(
        {
            "digit": predicted_digit,
            "confidence": confidence,
            "top_predictions": top_predictions,
            "processed_preview": image_array_to_data_url(preview_pixels),
        }
    )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
