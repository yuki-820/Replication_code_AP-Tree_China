```markdown
# Replication Code (Work in Progress)

This repository contains the replication code for the paper  
*"Factor Interaction Structure in the Chinese A‑Share Market: A Study Based on AP‑Tree"*.

**Note:** The project is still ongoing – some steps may be updated.

---

## Data Sources

The raw data used in this project originate from multiple sources:

- **Stock data** and **Q5factor** data are sourced from the **RESSET** database.  
- **CH3** and **CH4** factors are provided by **Mingshi Fund** – available for download at  
  [https://www.mingshiim.com/database](https://www.mingshiim.com/database).  
- **FF5** and **FF6** factors are obtained from the **CSMAR** database.

Pre‑computed factor files (derived from these sources) are already provided in `data/factors/` for convenience.

---

## Environment Setup

### Option 1 – Conda (recommended)

```bash
conda env create -f environment.yml
conda activate replication_env
```

### Option 2 – pip + venv

```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
# venv\Scripts\activate       # Windows
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Data Download

Raw monthly stock and financial data are available on Zenodo  
(DOI: 10.5281/zenodo.20032245).

Run the following to automatically download all files into `data/raw/`:

```bash
python download_data.py
```

Pre‑computed factor files are already provided in `data/factors/`.

---

## Run the Full Pipeline

```bash
python main.py
```

This sequentially executes all steps (see below).

---

## Steps (in order)

| Step | Script | Description |
|------|--------|-------------|
| 1 | `step1_data_process.py` | Preprocess raw data |
| 2 | `build_pool.py` (in each model folder) | Build candidate portfolios |
| 3 | `prune.py` (in each model folder) | Prune portfolios |
| 4 | `step4_factor_tests.py` | Factor model regressions and GRS tests |
| 5 | `step5_turnover.py` | Compute turnover and stock‑level weights |
| 6 | `step6_plot.py` | Generate figures |

You may also run each script individually, provided that the required inputs from previous steps exist.

---

## License

The code is released under the [MIT License](LICENSE).  
The raw data are provided under the same license.
```
