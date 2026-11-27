The training notebook downloads datasets directly. This folder is reserved for local cleaned/exported datasets.
PlantVillage: https://huggingface.co/datasets/mohanty/PlantVillage
India yield training source: https://huggingface.co/datasets/dhyann2815/india-crop-yield-prediction
Official crop production reference: https://www.data.gov.in/catalog/district-wise-season-wise-crop-production-statistics-0

Soil reference configuration:
- soil_reference_profiles.json keeps thresholds and crop profiles configurable.
- The only included numeric defaults are the Tamil Nadu Agricultural University (TNAU) available-N/P/K rating cutoffs, scoped to Tamil Nadu: https://agritech.tnau.ac.in/agriculture/agri_soil_soilratingchart.html
- TNAU's page does not state extraction method or elemental/oxide basis; verify a soil report matches before using those cutoffs.
- The same page's organic-carbon bands are internally inconsistent, so no organic-carbon cutoff is supplied. pH, moisture, EC, and crop suitability are also left unclassified until local references are configured.
- Add at least five locally verified crop profiles to enable top-five crop screening. The app deliberately does not seed guessed crop ranges or rankings.
