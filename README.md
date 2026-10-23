This repository is the codebase for bachelor thesis titled: 
# **Gated U-Net with Sentinel-1 and Sentinel-2 Fusion for Mapping Semeru 2021 and 2022 Eruption Materials: A Comparison of Random vs. Facies-Based Blocking Splits and GradientSHAP Feature Attribution**

**Author:** Susilo, D. D. R. (2026)<br>
**Institution:** Department of Geographic Information Science, Faculty of Geography, Universitas Gadjah Mada                                                                                

---

## Overview

This repository contains the complete preliminary computational pipeline for mapping volcanic materials (Pyroclastic Density Currents and Lahars) from the 2021 and 2022 Mount Semeru eruptions. 

The framework addresses three primary research objectives (RQs):
1. **Model Performance (RQ1):** Implementation of a dual-encoder Gated U-Net for adaptive SAR-Optical (Sentinel-1 & Sentinel-2) fusion.
2. **Spatial Validation (RQ2):** Structural diagnostics to separate spatial leakage from coverage problems by comparing Random Split against Facies-Based Blocking Split.
3. **Explainable AI (RQ3):** Feature attribution using GradientSHAP, including cross-seed and cross-facies stability testing to ensure the model learns physically meaningful features.

---

## Data Availability
All raw datasets used in this study are freely available from public repositories. To replicate this study, download the raw data from the sources below and place them in a `./data/raw/` directory.

*   **Sentinel-1 & Sentinel-2:** [Copernicus Data Space Ecosystem (CDSE)](https://dataspace.copernicus.eu/)
*   **Reference Labels (UNOSAT):** [UN-OCHA HDX](https://data.humdata.org/)
*   **DEM:** [Alaska Satellite Facility (ASF)](https://search.asf.alaska.edu/)

The specific dataset used in this study is available upon request.

##  Repository Structure

```text
semeru-unet-fusion-xai/
│
├── src/                        # Core Python modules
│   ├── config.py               # Global configuration & experiment registry
│   ├── data_loader.py          # Patch extraction, cloud masking, & split logic
│   ├── losses.py               # Focal Loss
│   ├── models.py               # Gated U-Net architecture
│   ├── train.py                # Training loop and ISR class weighting
│   ├── inference.py            # Full-AOI inference and Gaussian mosaicking
│   ├── evaluate_all_metrics.py # Full-AOI vs Test-only evaluation
│   ├── ensemble_maps.py        # 5-seed ensemble and uncertainty maps
│   ├── gradient_shap.py        # GradientSHAP spatial attribution
│   ├── shap_stability.py       # Attribution stability (Spearman, Facies Invariance)
│   └── leakage_analysis.py     # Structural diagnostics (Autocorrelation, Proximity)
│
├── requirements.txt            # Python dependencies
├── .gitignore                  # Excluded files (data, experiments, weights)
└── README.md
