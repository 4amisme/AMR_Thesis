# Spatiotemporal and Time Series Analysis of Multidrug-Resistant Bacteria Using National Surveillance Data from NARST

## Project Information

**Project Name:** Spatiotemporal and Time Series Analysis of Multidrug-Resistant Bacteria Using National Surveillance Data from NARST

**Project Number:** G40-HDS

**Program:** Health Data Science

**Department:** Computer Engineering

**Faculty:** Faculty of Engineering

**University:** King Mongkut's University of Technology Thonburi

**Academic Year:** 2025

**Project Members:**

* Ms. Chanokchon Karinruck
* Ms. Siraprapha Suriyasri

**Advisor:**

* Assoc. Prof. Peerapon Siripongwutikorn, Ph.D.

**Co-advisors:**

* Asst. Prof. Jiraphan Premsuriya, Ph.D.
* Pheerawas Chintrakulchai, Ph.D.

**GitHub Repository:**
https://github.com/4amisme/AMR_Thesis.git

---

## Project Overview

This project focuses on the analysis of antimicrobial resistance (AMR), especially multidrug-resistant (MDR) bacteria, using national surveillance data from the National Antimicrobial Resistance Surveillance Center, Thailand (NARST).

The project applies statistical analysis, time series forecasting, and spatiotemporal modeling to study the patterns, trends, and distribution of multidrug-resistant bacteria in Thailand. The analysis includes bacterial resistance trends, MDR pattern classification, model-based prediction, and spatial-temporal changes of MDR occurrence.

The results of this project can support antimicrobial resistance surveillance, public health planning, and data-driven decision-making.

---

## Objectives

1. To analyze patterns and trends of multidrug-resistant bacteria using time series analysis.
2. To study spatiotemporal changes of MDR bacteria across health regions and provinces in Thailand.
3. To analyze antimicrobial resistance trends between bacteria and specific antibiotics.
4. To classify MDR patterns and summarize antibiotic testing information.
5. To support visualization and interpretation of AMR surveillance data from NARST.

---

## Main Modules

The main source code is organized under the `MDR/` folder.

### 1. `MDR/model_LR`

This folder is used for AMR linear regression analysis.

The purpose of this module is to analyze the resistance trend between one bacterial organism and one selected antibiotic. This helps provide an overall view of how a specific bacterial species responds to a specific drug over time.

Main tasks include:

* Selecting bacteria-specific datasets
* Calculating prevalence of antimicrobial resistance
* Analyzing AMR trends by year
* Performing OLS and WLS regression
* Creating regional maps and trend outputs
* Summarizing model results for selected organisms

Example files and outputs include:

```text
(1.1) select data.py
(2.1) find prevalence.py
(2.2) find prevalence WLS.py
(2.3) region map.py
(3.1) OLS by year.py
(3.2) WLS.py
summary_models_*.csv
```

---

### 2. `MDR/DrugClass2`

This folder is used for MDR pattern processing and antibiotic class analysis.

The purpose of this module is to process antimicrobial susceptibility data and classify drug resistance patterns. It helps identify which MDR patterns exist in the dataset and how many antibiotic tests are available for each drug or drug class.

Main tasks include:

* Loading antimicrobial susceptibility data
* Processing MDR and XDR patterns
* Mapping drug class and ID information
* Counting antibiotic testing records
* Summarizing MDR patterns
* Creating charts for resistance pattern distribution

Example files include:

```text
(1) load data.py
(1-3) All process.py
(2.1) process_mdr.py
(2.2) process_xdr.py
(3) mapping ID.py
Drug_class_for_MDR_new.csv
ID.csv
pie chart.py
```

---

### 3. `MDR/model`

This folder is the main module for MDR time series prediction.

The purpose of this module is to forecast the trend of MDR patterns over time. The workflow includes calculating MDR percentages, checking seasonality, plotting seasonal patterns, and applying time series forecasting models.

Main tasks include:

* Calculating MDR pattern percentages
* Calculating MDR percentages by ward type
* Calculating MDR percentages by specimen type
* Checking ACF and seasonality strength
* Visualizing seasonal trends
* Running forecasting models
* Comparing model results

Forecasting models include:

* ARIMA
* SARIMA
* Simple Exponential Smoothing (SES)
* Triple Exponential Smoothing (TES)
* XGBoost

Example files include:

```text
(1.1) find % of MDR pattern.py
(1.2) find % by ward.py
(1.3) find % by specimen.py
(2) find ACF & strength.py
(3) Seasonal Graph.py
(4) SARIMA.py
(5) ARIMA.py
(6) TES.py
(7) SES.py
(8) XGBoost.py
(9) All model.py
```

Output summary files include:

```text
all_bacteria_seasonality_summary.csv
drug_resistance_patterns_summary.csv
Spec_type_seasonality_summary.csv
Ward_type_seasonality_summary.csv
```

---

### 4. `MDR/spatiotemperal`

This folder is used for spatial and spatiotemporal analysis of MDR bacteria.

The purpose of this module is to analyze changes in MDR resistance across both space and time. It includes regional and provincial data preparation, spatial analysis, INLA-based spatiotemporal modeling, WAPE evaluation, and map visualization.

Main tasks include:

* Calculating MDR resistance percentages by region
* Preparing province-level and region-level datasets
* Adding seasonal variables
* Running Bayesian spatiotemporal models
* Forecasting MDR trends by area
* Creating spatial prediction maps
* Evaluating prediction performance using WAPE

Example files include:

```text
(1) find % by region.py
(1.1) find %R with province.py
(2.1)_add_season.ipynb
(3) add_season_for_R.ipynb
INLA_R_percent.R
beta_R_inla.R
gaussian_R_inla.R
spatio-temp.R
wape_R_inla.R
```

Example output files include:

```text
aba_Forecast_5Years_All_Patterns_LogitGaussian.csv
eco_Forecast_5Years_All_Patterns_LogitGaussian.csv
pae_Forecast_5Years_All_Patterns_LogitGaussian.csv
*_Map_Pattern*_2025_2029.png
```

> Note: The folder name in the repository is currently written as `spatiotemperal`.

---

## Repository Structure

```text
AMR_Thesis/
├── .vscode/
├── MDR/
│   ├── DrugClass2/
│   ├── model/
│   ├── model_LR/
│   └── spatiotemperal/
├── notebooks/
├── references/
├── scripts/
├── src/
├── README.md
├── requirements.txt
└── file_row_counts_summary.csv
```

---

## Technologies Used

### Programming Languages

* Python
* R
* Jupyter Notebook

### Python Libraries

The main Python dependencies listed in `requirements.txt` include:

```text
pandas>=2.2
numpy>=1.26
openpyxl>=3.1
tqdm>=4.66
python-dateutil>=2.9
```

Additional libraries may be required depending on the script being executed, such as:

```text
matplotlib
seaborn
scikit-learn
statsmodels
xgboost
geopandas
```

### R Libraries

Some spatiotemporal analysis scripts require R. Depending on the model script, additional R packages may be required, such as:

```text
INLA
dplyr
ggplot2
sf
sp
readr
```

---

## Installation Manual

### 1. Clone the Repository

```bash
git clone https://github.com/4amisme/AMR_Thesis.git
cd AMR_Thesis
```

---

### 2. Create a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```
---

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

If some scripts require additional packages, install them manually:

```bash
pip install matplotlib seaborn scikit-learn statsmodels xgboost geopandas
```

---

### 4. Install R and Required R Packages

Some files in `MDR/spatiotemperal/` require R for INLA-based spatiotemporal modeling.

Please install R and RStudio, then install the required packages in R:

```r
install.packages(c("dplyr", "ggplot2", "sf", "sp", "readr"))
```

For INLA, install it using:

```r
install.packages(
  "INLA",
  repos = c(getOption("repos"), INLA = "https://inla.r-inla-download.org/R/stable"),
  dep = TRUE
)
```

---

## Data Preparation

The original NARST dataset is not included in this repository because it contains confidential surveillance data.

Before running the scripts, please prepare the required input files and check the file paths inside each script. Some scripts may require CSV or Excel files generated from previous preprocessing steps.

The general workflow is:

```text
Raw NARST data
→ Data cleaning
→ WHONET processing
→ MDR pattern classification
→ Time series dataset preparation
→ Spatiotemporal dataset preparation
→ Model training and evaluation
```

---

## How to Run the Project

### 1. MDR Pattern Processing

Run scripts in:

```text
MDR/DrugClass2/
```

Suggested order:

```text
(1) load data.py
(2.1) process_mdr.py
(2.2) process_xdr.py
(3) mapping ID.py
```

or use:

```text
(1-3) All process.py
```

---

### 2. AMR Linear Regression Analysis

Run scripts in:

```text
MDR/model_LR/
```

Suggested order:

```text
(1.1) select data.py
(2.1) find prevalence.py
(2.2) find prevalence WLS.py
(2.3) region map.py
(3.1) OLS by year.py
(3.2) WLS.py
```

This module analyzes the resistance trend between selected bacteria and selected antibiotics.

---

### 3. MDR Time Series Prediction

Run scripts in:

```text
MDR/model/
```

Suggested order:

```text
(1.1) find % of MDR pattern.py
(1.2) find % by ward.py
(1.3) find % by specimen.py
(2) find ACF & strength.py
(3) Seasonal Graph.py
(4) SARIMA.py
(5) ARIMA.py
(6) TES.py
(7) SES.py
(8) XGBoost.py
(9) All model.py
```

This module performs MDR trend forecasting using time series models.

---

### 4. Spatiotemporal Analysis

Run scripts and notebooks in:

```text
MDR/spatiotemperal/
```

Suggested workflow:

```text
(1) find % by region.py
(1.1) find %R with province.py
(2.1)_add_season.ipynb
(3) add_season_for_R.ipynb
INLA_R_percent.R
spatio-temp.R
wape_R_inla.R
```

This module analyzes spatial and temporal changes of MDR bacteria and produces forecast results and map visualizations.

---

## Model Evaluation

The project uses error metrics such as Weighted Absolute Percentage Error (WAPE) to evaluate forecasting performance.

WAPE is used to compare predicted values with actual values and summarize the overall percentage error of the forecasting model.

---

## Outputs

The project produces several types of outputs, including:

* Cleaned and processed AMR datasets
* MDR pattern summary files
* Antibiotic testing count summaries
* AMR trend analysis results
* OLS and WLS regression summaries
* Time series forecasting results
* Seasonality summary files
* Spatiotemporal prediction files
* Provincial or regional map visualizations
* WAPE evaluation results

---

## Important Notes

* The original NARST dataset is not included in this repository due to privacy and confidentiality restrictions.
* Some scripts require local file paths, so users may need to update paths before running.
* Some outputs depend on previous processing steps.
* R and INLA are required for some spatiotemporal modeling scripts.
* The folder `MDR/spatiotemperal/` contains the spatiotemporal analysis files, although the folder name is spelled `spatiotemperal`.

---

## License

This project is developed for academic purposes as part of a final-year project in Health Data Science. Please contact the project members or advisors before using or distributing the source code.

---

## Contact

For more information, please contact the project members.
