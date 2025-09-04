# deepchem-DEL
DeepChem-DEL Infrastructure


## KinDEL Data Overview

For our experiments (as described in the paper), we use the 1M datapoint subsets of the KinDEL datasets provided by the KinDEL team.

KinDEL (Kinase Inhibitor DNA-Encoded Library) is a large-scale DNA-Encoded Library (DEL) dataset designed for kinase inhibitor discovery. The full dataset contains ~81 million compounds per kinase target, but for reproducibility and efficiency, we restrict our work to the 1 million datapoint samples curated for two kinase targets:

MAPK14 (mapk14_1M.parquet)

DDR1 (ddr1_1M.parquet)

In addition, we evaluate models on heldout datasets (heldout/) to measure generalization.

## Folder Structure

```
DEEPCHEM-DEL/
│
├── data_preprocessing/
│   ├── data/
|       ├──training/
│       │  ├── mapk14_1M.parquet
│       │  └── ddr1_1M.parquet
│       │
│       └── heldout/
│           ├── ddr1_offdna.csv
│           ├── ddr1_ondna.csv
│           ├── mapk14_offdna.csv
│           ├── mapk14_ondna.csv
│
├── src/

```