# Dataset

This folder contains the gene expression dataset used for cancer classification.

## Files Required
- `data.csv` - Gene expression data (801 samples × 20,531 genes)
- `labels.csv` - Cancer type labels for each sample (5 classes: BRCA, COAD, KIRC, LUAD, PRAD)

## Download Instructions

The dataset is sourced from **TCGA (The Cancer Genome Atlas)** via Kaggle.

**To run this project:**
1. Download the dataset from Kaggle: https://www.kaggle.com/datasets/waalbannyantudre/gene-expression-cancer-rna-seq-donated-on-682016
2. Extract the files
3. Place `data.csv` and `labels.csv` in this directory
4. Run the Jupyter notebook in `notebooks/cancer_analysis.ipynb`

**Please Note:** 
Due to GitHub's file size limitations (data.csv > 100MB), the raw data files are not included in this repository.
