# AgriSmart

AgriSmart is a Streamlit application that brings together AI-assisted crop disease detection, soil-photo classification, crop-yield estimates, weather-aware farming guidance, and an agriculture chatbot.

## Features

- Farmer registration, sign-in, profile, and local activity history.
- Crop disease detection and crop-specific fertilizer suggestions.
- Soil-photo classification with sourced general crop guidance and PDF reporting.
- Crop-yield prediction when the separately generated yield model is installed.
- Local weather forecasts and farming advice using Open-Meteo.
- AgriBot answers from built-in FAQs and local knowledge; optional Groq-powered answers, multilingual responses, and voice transcription.

Soil photos do not measure soil nutrients, pH, moisture, or salinity. App results are educational guidance, not a confirmed diagnosis or a substitute for local agricultural advice.

## Technologies

Python 3.11, Streamlit, TensorFlow/Keras, NumPy, pandas, scikit-learn, Pillow, SQLite, Requests, and the supporting packages listed in `requirements.txt`.

## Project structure

```text
AgriSmart_AI_Project/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml.example
├── backend/                 # App features, model loading, API services, and reports
├── data/                    # Small JSON configuration and soil guidance
├── database/                # Local users.db is created when the app starts; not committed
├── knowledge_base/          # Local answer excerpts, source manifest, and index
├── models/                  # Small model metadata plus disease and soil classifiers
├── notebooks/               # Google Colab training notebooks
├── static/                  # App images
└── tests/                   # unittest test suite
```

## Prerequisites

- Git.
- Python 3.11.
- Internet access for installation, weather requests, and optional Groq API requests.

TensorFlow 2.15 or newer may not install on every native Windows/Python combination. If TensorFlow installation fails on Windows, use a supported Linux environment such as WSL 2, then follow the Linux commands below.

## Clone and run on Windows (PowerShell)

Replace `OWNER/REPOSITORY` with the GitHub repository path. Run these commands from the parent folder where you want the project:

```powershell
git clone https://github.com/OWNER/REPOSITORY.git
Set-Location .\REPOSITORY
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, allow scripts for this terminal session and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Linux / WSL 2

Run from the cloned repository folder:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`) when the app starts.

## Models and datasets

The disease and soil classifier model files and their required labels/metadata are included in `models/`. They are each about 21 MB. The source notebooks and training datasets are not bundled as raw data; the Colab notebooks retrieve their datasets from their documented sources.

`models/yield_model.joblib` is about 164 MB and is intentionally excluded because GitHub rejects individual files larger than 100 MB. Yield prediction needs this file. To create it:

1. Open `notebooks/AgriSmart_Colab_Training.ipynb` in Google Colab and run the cells in order (a GPU runtime is recommended).
2. Download and extract the generated `AgriSmart_Models.zip`.
3. Copy `yield_model.joblib` into this project's `models/` folder.

The `.gitignore` keeps this local file out of Git. The same notebook downloads PlantVillage (about 2.2 GB) and an India crop-yield dataset for training; those datasets are not needed to run the bundled disease and soil models. Soil classifier retraining instructions are in `notebooks/Soil_Image_Classification_Colab.ipynb`.

## Optional Groq API configuration

The API key is optional. Without it, AgriBot can still use built-in FAQs and local knowledge excerpts in English. A key enables open-ended answers, multilingual responses, and voice transcription.

For Streamlit secrets, copy the safe template and replace its placeholder locally:

```powershell
Copy-Item .streamlit\secrets.toml.example .streamlit\secrets.toml
```

Edit `.streamlit/secrets.toml` to contain:

```toml
GROQ_API_KEY = "your-groq-api-key"
```

Alternatively, set the key in the same PowerShell window used to run Streamlit:

```powershell
$env:GROQ_API_KEY = "your-groq-api-key"
streamlit run app.py
```

Never commit `.streamlit/secrets.toml`, API keys, or other credentials. The real secrets file and common `.env` files are ignored by Git.

## Run the tests

From the repository root with the virtual environment activated:

```powershell
python -m unittest discover -s tests
```

## Troubleshooting

- **`streamlit` is not recognized:** activate `.venv` and run `python -m pip install -r requirements.txt` again; you can also start with `python -m streamlit run app.py`.
- **`ModuleNotFoundError: backend`:** run the command from the repository root, the folder containing `app.py`.
- **TensorFlow installation or import fails on Windows:** use WSL 2/Linux with a supported Python/TensorFlow combination.
- **Yield prediction reports a missing model:** generate `yield_model.joblib` with `notebooks/AgriSmart_Colab_Training.ipynb` and place it in `models/`.
- **AgriBot cannot answer open-ended questions or transcribe audio:** configure a valid `GROQ_API_KEY`; built-in FAQs and local excerpts remain available without one.

## Data sources and limitations

- Plant disease training source: [PlantVillage on Hugging Face](https://huggingface.co/datasets/mohanty/PlantVillage), associated with Mohanty, Hughes & Salathé (2016), DOI [10.3389/fpls.2016.01419](https://doi.org/10.3389/fpls.2016.01419). Review and follow the dataset license and attribution requirements.
- Soil image source: [Soil Classification Dataset](https://github.com/Phantom-fs/Soil-Classification-Dataset), DOI [10.1016/j.engappai.2025.111425](https://doi.org/10.1016/j.engappai.2025.111425). The selected images are licensed CC BY-NC 4.0; do not use them commercially without permission. The seven labels are Alluvial, Arid, Black, Laterite, Mountain, Red, and Yellow Soil.
- Yield training source: [India crop-yield dataset on Hugging Face](https://huggingface.co/datasets/dhyann2815/india-crop-yield-prediction). Validate its provenance, units, and labels before relying on its estimates.
- Weather: [Open-Meteo](https://open-meteo.com/en/terms). Forecasts provide short-range context, not seasonal climate normals.
- AgriBot's included knowledge excerpts are limited and are not comprehensive agricultural advice.

Plant disease models trained on controlled PlantVillage images may perform worse on field photographs. Soil model confidence is not a calibrated probability, and soil-photo classification cannot measure laboratory properties. Yield outputs are estimates. Verify important decisions with locally validated information and an agricultural expert.
