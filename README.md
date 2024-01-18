# A tutorial for generating and evaluating synthetic health data based on MIMIC-IV V2.0 dataset and EMR-WGAN.
This repository is paired with the following tutorial paper:

*Yan C, Zhang Z, Nyemba S, Li Z. Generating synthetic electronic health record data using generative adversarial networks: A tutorial.*

## System requirement

### OS Requirements
This package is supported for *Linux*. The package has been tested on the following systems:
+ Linux: Ubuntu 20.04

### Python (3.7.16) Dependencies

```
tensorflow
pandas
numpy
pandas
argparse
matplotlib
scipy
sklearn
lightgbm
joblib
shap
requests
```
### Install all Python dependencies
```
pip install -r requirements.txt
```

## Descriptions of files

### Data preprocessing
Before this step, one needs to download MIMIC-IV V2.0 data at https://physionet.org/content/mimiciv/2.0/ by completing the required application and training steps.

Please run `data_extraction_and_preprocessing_github.ipynb file` step by step to prepare the dataset for the subsequent GAN training.

### GAN training  
Train an EMR-WGAN model by specifying gpu_id and model_id and then running
```
python GAN_training.py --gpu_id xx --model_id xx
```

### Synthetic data generation
Generate synthetic data from a trained EMR-WGAN model by specifying gpu_id, model_id, and load_checkpoint (ie, checkpoint id).
```
python GAN_generation.py --gpu_id xx --model_id xx --load_checkpoint xx
```


## Reference to cite

*Yan C, Yan Y, Wan Z, Zhang Z, Omberg L, Guinney J, Mooney SD, Malin BA. A multifaceted benchmarking of synthetic electronic health record generation models. Nature communications. 2022 Dec 9;13(1):7609.*

*Zhang Z, Yan C, Mesa DA, Sun J, Malin BA. Ensuring electronic medical record simulation through better training, modeling, and evaluation. Journal of the American Medical Informatics Association. 2020 Jan;27(1):99-108.*
