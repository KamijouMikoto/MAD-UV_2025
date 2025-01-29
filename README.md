# MADUV Challenge 2025 Baseline Code

This repository provides the baseline code for **The 1st INTERSPEECH Mice Autism Detection via Ultrasound Vocalisation 
(MADUV) Challenge**, focusing on the classification of ultrasonic vocalisations (USVs) from mice.

The baseline code demonstrates signal preprocessing, feature extraction, model training, validation, and prediction.

More information about the challenge via https://www.maduv.org/.

---

## Overview  

This code is designed to provide a starting point to participants in the challenge. It includes several steps described 
in the baseline paper:

- **Trimming**: Trimming the audio files into segments of 30s with an overlap of 15s.
- **Feature Extraction**: Extracting *full* (full spectrogram), *ultra* (ultrasonic spectrogram) and *audi* (audible 
spectrogram).  
- **Model Training**: A two-layer CNN for classification.
- **Prediction**: Predicting the class of audio samples in the test set.

## File Structure
```
/MADUV_2025/
│
├── trim.py                     # Script for trimming audio files
├── feature_extractor.py        # Script for extracting features
├── model.py                    # Script for model architecture
├── data_pipeline.py            # Script for preparing data
├── hyperparameter.py           # Experimental hyperparameters and settings
├── train.py                    # Script for training the model
├── test.py                     # Script for predicting samples
├── requirements.txt            # List of dependencies
└── README.md                   # Documentation
```

## Usage (Reproduction)

### 1. Dependencies Installation

First, install the required dependencies included in ```requirements.txt```.

### 2. Data Trimming

Run the following command to trim the audio files into segments of 30s with a 15s overlap:

```bash
python trim.py --input_path <path_to_dataset> --output_path <path_to_output_dir> --chunk 30000 --overlap 15000
```

### 3. Feature Extraction

Run the following command to extract *full*, *ultra* and *audi* features from the audio files:

```bash
python feature_extractor.py --input_path <path_to_trimmed_dir> --output_path <path_to_output_dir> --feature_set <feature_set>
```

### 4. Model Training

Set the hyperparameters and settings in ```hyperparameter.py```. And then, run the following command to train the model:

```bash
python train.py
```

### 5. Prediction

Run the following command to predict the class of audio samples in the test set:

```bash
python test.py
```

## Citation

If you use this baseline in your research, please cite:

```bibtex
@Article{zYang25-MAD,
  author =  "Z. Yang and M. Song and X. Jing and H. Zhang and K. Qian and B. Hu and K. Tamada and T. Takumi and B. W. Schuller and Y. Yamamoto",
  title =   "{MADUV}: The 1st {INTERSPEECH} Mice Autism Detection via Ultrasound Vocalization Challenge",
  journal = "arXiv preprint arXiv:2501.04292",
  year =    "2025"
}
```

## Contact
For any questions or issues, please contact the organising team of MADUV Challenge:

Email: info@maduv.org or zijiang.yang@ieee.org
