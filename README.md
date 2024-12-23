# Investigation of Mitochondrial and Viability Phenotypes in Fibroblast-Derived Motor Neurons
This repository contains the code and analysis pipelines accompanying the paper:

Investigation of mitochondrial and viability phenotypes in motor neurons derived by direct conversion of fibroblasts from familial ALS subjects
Running title: Mitochondria in ALS fibroblast-derived motor neurons
Authors: Csaba Konrad, Evan Woo, Faiza Tasnim, Hibiki Kawamata, Giovanni Manfredi

Table of Contents
Overview
Repository Structure
Dependencies
Data Preparation
Usage
Reproducing the Analyses
Citation
Contact
License
Overview
We employed a direct conversion (DC) technique to generate induced motor neurons (iMN) from familial ALS (fALS) and control fibroblasts. The code in this repository supports:

Deep learning-based cell classification:
A ResNet-50 pipeline that classifies cells as dead, non-neuron, or iMN in various imaging modalities (e.g., hSyn-eGFP live imaging, TDP-43 immunocytochemistry, TMRM live imaging).
A YOLOv8-based pipeline to detect and track single iMN over time (longitudinal survival studies).
Automated image analysis and segmentation:
CellProfiler scripts and ImageJ macros for morphological measurements and single-cell segmentation.
Mitochondrial membrane potential and motility analyses:
Scripts for reading TMRM live imaging data, segmenting individual mitochondria, skeletonizing, and computing morphological and motility parameters.
Bioenergetic data analysis:
Python notebooks for importing Seahorse (XF) data, normalizing for cell number, and calculating OCR/ECAR parameters.
Scripts for analyzing PercevalHR sensor data (ATP/ADP ratio).
Viability analyses:
Kaplan-Meier and Cox proportional hazards modeling using the lifelines Python package.
The main goals of these scripts/notebooks are to automate, standardize, and reproduce the analyses described in the paper.

Repository Structure
Below is a general outline of the repository structure. File and folder names may vary depending on your local clone.

bash
Copy code
.
├── data/                    
│   ├── example_images/            # Example image sets for testing the pipelines
│   ├── seahorse_data/            # Example XF data for OCR/ECAR calculations
│   └── ...
├── notebooks/
│   ├── 01_iMN_classification.ipynb  # Example Jupyter notebook for ResNet-50 classification
│   ├── 02_YOLOv8_detection.ipynb    # YOLOv8-based iMN detection & tracking
│   ├── 03_survival_analysis.ipynb   # Survival analysis (Kaplan-Meier, Cox)
│   ├── 04_mito_analysis.ipynb       # Mitochondrial morphology, TMRM, motility
│   ├── 05_seahorse_analysis.ipynb   # OCR/ECAR data import and processing
│   ├── 06_percevalHR_analysis.ipynb # ATP/ADP ratio analysis
│   └── ...
├── src/
│   ├── models/                      # Trained models (ResNet-50, YOLOv8 weights)
│   ├── utils/
│   │   ├── image_preprocessing.py   # Functions for background subtraction, thresholding, etc.
│   │   ├── classification_utils.py  # Helper functions for model inference
│   │   ├── tracking_utils.py        # BoT-SORT or other tracking methods
│   │   └── ...
│   └── ...
├── environment.yml                  # Conda environment file (recommended packages/versions)
├── requirements.txt                 # Alternatively, pip install requirements
├── README.md                        # This file
└── LICENSE                          # License for usage
Dependencies
We recommend using conda or a Python virtual environment to manage dependencies:

Python >= 3.8
PyTorch >= 1.12
torchvision (for ResNet-50 or custom CNN inference)
Ultralytics YOLOv8 (for object detection and tracking)
OpenCV (computer vision operations)
lifelines (Kaplan-Meier, Cox PH survival analysis)
numpy, pandas, scikit-learn, matplotlib, seaborn, scipy
CellProfiler (optional, for batch segmentation)
ImageJ/Fiji (optional, macros for custom segmentation steps)
Install with either:

bash
Copy code
# Using conda environment
conda env create -f environment.yml
conda activate als-imn-env
or

bash
Copy code
# Using pip
pip install -r requirements.txt
Data Preparation
Imaging Data

Store raw images (TIFF, PNG, etc.) in the data/example_images/ directory or specify a custom path in your notebooks.
For TMRM, PercevalHR, or immunocytochemistry images, ensure the relevant channels (e.g., Hoechst, TMRM, GFP, or TDP-43) are available in separate files or combined multi-channel TIFFs.
Seahorse Data

Place raw or exported Seahorse files (Excel/CSV) into data/seahorse_data/.
The analysis scripts assume a standard Seahorse file format. You may need to adjust column names in 05_seahorse_analysis.ipynb depending on your data export.
Annotation Files for YOLOv8

If you are retraining YOLOv8 for cell detection, place your bounding box annotation files (COCO or YOLO format) in data/annotations/.
Trained Models

Pretrained weights for ResNet-50 or YOLOv8 are stored in src/models/. Check the exact filenames used in the notebooks.
If you wish to train your own models, you will need to prepare a dataset of labeled cell images.
Usage
ResNet-50 Classification

Open notebooks/01_iMN_classification.ipynb.
Update the data_dir and model_path variables to point to your images and model files.
Run all cells to perform classification (dead, non-neuron, or iMN).
YOLOv8 Detection and Tracking

Open notebooks/02_YOLOv8_detection.ipynb.
Modify paths to images and YOLOv8 weights.
Run for detection and subsequent tracking with BoT-SORT or other integrated trackers.
Survival Analysis

Open notebooks/03_survival_analysis.ipynb.
This notebook shows how we generate Kaplan-Meier curves and Cox PH models using lifelines.
Mitochondrial Analysis

Open notebooks/04_mito_analysis.ipynb.
Demonstrates TMRM segmentation, mitochondria morphology quantification, and motility (kymograph) extraction.
Seahorse and PercevalHR

05_seahorse_analysis.ipynb: Imports OCR/ECAR data, normalizes by cell count, calculates basal and maximal respiration, glycolytic parameters.
06_percevalHR_analysis.ipynb: Processes time-lapse images of PercevalHR-labeled cells to extract ATP/ADP ratios.
Reproducing the Analyses
Clone this repository:
bash
Copy code
git clone https://github.com/YourUsername/ALS-iMN-mitochondria.git
cd ALS-iMN-mitochondria
Set up environment (conda or pip):
bash
Copy code
conda env create -f environment.yml
conda activate als-imn-env
or
bash
Copy code
pip install -r requirements.txt
Open the notebooks in JupyterLab or Jupyter Notebook:
bash
Copy code
jupyter lab
Run the notebooks in order:
01_iMN_classification.ipynb
02_YOLOv8_detection.ipynb
03_survival_analysis.ipynb
04_mito_analysis.ipynb
05_seahorse_analysis.ipynb
06_percevalHR_analysis.ipynb
Adjust parameters (paths, hyperparameters, etc.) as needed.

Citation
If you use this code in your research, please cite:

Konrad C, Woo E, Tasnim F, Kawamata H, Manfredi G.
Investigation of mitochondrial and viability phenotypes in motor neurons derived by direct conversion of fibroblasts from familial ALS subjects. (Year).

BibTeX entry (placeholder):

bibtex
Copy code
@article{Konrad202X,
  title={Investigation of mitochondrial and viability phenotypes in motor neurons derived by direct conversion of fibroblasts from familial ALS subjects},
  author={Konrad, Csaba and Woo, Evan and Tasnim, Faiza and Kawamata, Hibiki and Manfredi, Giovanni},
  journal={Neurobiology of Disease/Cell Reports/Etc},
  year={202X},
  volume={XX},
  pages={YY--ZZ}
}
Contact
For questions, please contact:

Dr. Csaba Konrad
Feil Family Brain and Mind Research Institute
Weill Cornell Medicine
407 East 61st Street, RR511
New York, NY 10065
Phone: 646-962-8271
Email: csk2001@med.cornell.edu

License
This project is licensed under the MIT License. Feel free to use, modify, and distribute this code. Please acknowledge our work by citing the above publication when using any part of this repository.
