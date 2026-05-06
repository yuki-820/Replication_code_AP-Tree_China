# Replication Code for "Factor Interaction Structure in the Chinese A‑Share Market: A Study Based on AP‑Tree"

---

## Repository Structure

Replication_Code/
├── code/                                 # All Python scripts
│   ├── step1_data_process.py             # Step 1: Data preprocessing
│   ├── step4_factor_tests.py             # Step 4: Factor model regressions & GRS
│   ├── step5_turnover.py                 # Step 5: Turnover analysis
│   ├── TripleSort64_full/                # Model 1 – build & prune
│   ├── TripleSort128_cleaned/            # Model 2 – build & prune
│   ├── AP-Tree_full/                     # Model 3 – build & prune
│   ├── AP-Tree_cleaned/                  # Model 4 – build & prune
│   ├── TripleSort128_cleaned_longonly/   # Model 5 – prune only
│   └── AP-Tree_cleaned_longonly/         # Model 6 – prune only
├── data/                                 # Data directory
│   ├── raw/                              # Raw data (auto‑downloaded)
│   │   ├── monthly/                      # Monthly stock CSV files
│   │   └── financial/                    # Balance sheet & income statement CSVs
│   ├── factors/                          # Pre‑computed factor returns (provided)
│   │   └── (including factor files, variable description)
│   └── year/                             # Step 1 output (processed yearly files) (2011‑2025)
├── output/                               # All results (auto‑created)
│   ├── candidate_pools/                  # Step 2 output (candidate portfolios)
│   ├── pruned/                           # Step 3 output (pruned weights & returns)
│   ├── tables/                           # Step 4 output (paper tables in CSV)
│   └── stock_weights/                    # Step 5 output (stock‑level weights & turnover summary)
├── download_data.py                      # Script to fetch raw data from Zenodo
├── main.py                               # Master pipeline (automates all steps)
├── requirements.txt                      # Pip dependencies
├── environment.yml                       # Conda environment
└── LICENSE                               # MIT license

## Environment Setup

Prerequisites
Python 3.9 or higher (recommended: 3.11)

pip or conda package manager

### Option 1: Using Conda (recommended for full environment reproducibility)

Create and activate the environment from the environment.yml file:

```
bash

# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate replication_env

# (Optional) Verify that all packages are correctly installed
conda list
```

If you encounter dependency conflicts, try:

```
bash

conda env update -f environment.yml --prune
```

### Option 2: Using pip + venv (lightweight alternative)

Create a virtual environment and install dependencies from requirements.txt:

```
bash

# Create a virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Upgrade pip (optional but recommended)
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

Verify the Environment
Run the following Python code to confirm all key libraries are correctly installed:

```
python

import sys
import numpy as np
import pandas as pd
import sklearn
import statsmodels
import scipy
import openpyxl
import matplotlib
import tqdm

print(f"Python: {sys.version}")
print(f"numpy: {np.__version__}")
print(f"pandas: {pd.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"statsmodels: {statsmodels.__version__}")
print(f"scipy: {scipy.__version__}")
print(f"openpyxl: {openpyxl.__version__}")
print(f"matplotlib: {matplotlib.__version__}")
print(f"tqdm: {tqdm.__version__}")
Expected output (versions may vary slightly):

text

Python: 3.11.11 | packaged by Anaconda, Inc. ...
numpy: 1.26.4
pandas: 2.2.3
scikit-learn: 1.6.1
statsmodels: 0.14.4
scipy: 1.13.1
openpyxl: 3.1.5
matplotlib: 3.10.1
tqdm: 4.66.5
If any package is missing, install it manually with pip install <package_name>.
```

---

## Data Download

### Raw Monthly & Financial Data 

The raw monthly stock and financial statement data are publicly available on Zenodo
(DOI: 10.5281/zenodo.20032245).

Run the following command to automatically download all files into data/raw/:

```
bash

python download_data.py
```

### Factor Data

The pre‑computed factor return files (CH3, CH4, FF5, FF6, Carhart4) are already provided in the `data/factors/` directory:

- `CH3_factors_monthly_202602.xlsx`
- `CH4_factors_monthly_202602.xlsx`
- `fivefactor_monthly.csv`
- `fivefactor_monthly_variable_note.txt`
- `sizevaluechina(section5-CH3,section7-CH4).pdf` (which references the exact section numbers where factor construction is described)

No further action is required for these files. The pipeline will read them automatically during the factor tests.

---

## Running the Replication

### Fully Automated Run

```
bash

python main.py
```

`main.py` sequentially executes all steps:

1. Data preprocessing (`step1_data_process.py`)
2. Building candidate pools for all six models (each `build_pool.py`)
3. Pruning (each `prune.py`)
4. Factor model tests (`step3_factor_tests.py`)
5. Turnover analysis (`step4_turnover.py`)

### Manual / Step‑by‑Step Execution

You may also run each script individually, as long as the required inputs (e.g., candidate pool files from `build_pool.py`) exist. For example:

```
bash

# Step 1: Preprocess data
python code/step1_data_process.py

# Step 2: Build candidate pools for a specific model
python code/TripleSort64_full/build_pool.py

# Step 3: Prune a specific model
python code/TripleSort64_full/prune.py

# Step 4: Factor tests
python code/step3_factor_tests.py

# Step 5: Turnover analysis
python code/step4_turnover.py
```

You can also modify parameters inside the individual scripts (e.g., `kmin`, `kmax`, `lambda0_list`, `train_end_date`) before running them.

---

## Parameter Configuration

Each model folder (e.g., `TripleSort64_full/`) contains its own configuration inside `build_pool.py` and `prune.py`. To change model‑specific settings (e.g., number of nodes for AP‑Tree, LARS hyperparameters), edit the corresponding script.
Global paths are defined relative to `PROJECT_ROOT`, so no absolute path modification is required.

---

## Output Tables

All final tables are saved in `output/tables/` as CSV files, including:

* `APTree_GRS_original.csv`
* `TripleSort64_top10_CH4_alpha.csv`
* `APTree_K20_top10_CH4_alpha.csv`
* `APTree_K40_top10_CH4_alpha.csv`
* `APTree_clean_K20_top8_alpha.csv`
* `APTree_clean_K40_top8_alpha.csv`
* `LongOnly_Comparison.csv`
* `GRS_Cleaned_Methods.csv`
* `All_Alpha_Detailed.csv`

---

## Turnover Analysis Output

The turnover analysis script (`step5_turnover.py`) produces stock‑level weights and average monthly one‑way turnover for each cleaned‑sample strategy (AP-Tree K=20, K=40, and TripleSort128). Results are saved in:

- `output/stock_weights/` – contains final weight matrices for each section and the turnover summary:
 `AP20_cleaned_Sec*_weights.csv`
 `AP40_cleaned_Sec*_weights.csv`
 `Triple128_cleaned_Sec*_weights.csv`
 `Average_Turnover_Summary_Cleaned.csv`
- `output/mappings/` – intermediate stock‑to‑node mapping files (automatically generated, can be deleted after run)

## License

The code is released under the MIT License. The raw data are provided under the same license (MIT).
