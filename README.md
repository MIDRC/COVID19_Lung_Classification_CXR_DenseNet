# Stanford COVID-19 Diagnosis on X-Rays (MIDRC CRP-4)

**Modality:** Chest X-ray

COVID-19 Diagnosis based on a DenseNet architecture. Models were trained on an artificial mix of data from RICORD, CheXpert and more. The preprocessing steps involve lung segmentation based on the [lungVAE network](https://github.com/raghavian/lungVAE). The model is built using PyTorch Lightning.

**Requirements:** Pytorch, Pytorch Lightning, Pydicom

## Preprocessing
```
preprocessing.py --help
--ricord PREPROCESS RAW RICORD
--chexpert PREPROCESS RAW CHEXPERT
--iitac PREPROCESS RAW IITAC
--negative_sample_size NEGATIVE SAMPLE SIZE USED BY CHEXPERT (e.g. 5000)
--data_dir PATH TO INPUT FILES
--output_dir PATH TO OUTPUT FILES
--device DEVICE (e.g., cuda:1)
```

## Training
```
train.py --help
--learning_rate LEARNING_RATE
--backbone BACKBONE
--freeze FREEZE ALL LAYERS BUT CLASSIFIER
--aux_lambd AUXILLIARY LOSSES
--clf_features NUMBER OF FEATURES OF HIDDEN LAYER IN CLASSIFIER
--weights WEIGHTS
```

## Predict

```
predict.py --help
--ckpt PATH TO CHECKPOINT (e.g., output/name_of_checkpoint.pt)
--output_path PATH TO OUTPUT CSV
```
