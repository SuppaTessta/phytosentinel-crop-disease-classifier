# PhytoSentinel

Crop disease classifier for leaf photos: ResNet50 (ImageNet) fine-tuned on the PlantVillage dataset (38 classes), with a Gradio demo that shows the predicted class, its confidence and a Grad-CAM heatmap.

## Results (held-out test set)
ResNet50 fine-tuned on PlantVillage `color` (38 classes, 54,305 images) on a Kaggle Tesla T4 in `notebooks/04_kaggle_finetune.ipynb`.

| Metric (test, n = 8,146) | Value |
|---|---|
| Accuracy | 98.53% |
| Macro-F1 | 0.981 |
| Weighted-F1 | 0.985 |
| Top-5 accuracy | 99.95% |

- Split: stratified 70/15/15 (38,013 / 8,146 / 8,146), seed 42. The test split is used once, after training.
- Training: 5 epochs with the base frozen (head only, Adam 1e-3), then 10 epochs with the `conv5` block unfrozen (Adam 1e-5, BatchNorm frozen). Early stopping on val loss kept the best epoch (val accuracy 98.42%). Mixed precision, class weights, augmentation on train only, `resnet50.preprocess_input`.
- Wall time: 63 min on one T4 (TensorFlow 2.21).
- Weakest classes (test F1): Corn Cercospora/Gray leaf spot 0.90, Potato healthy 0.91, Tomato Target Spot 0.93.
- Artifacts: `results/metrics.json` (with per-class report), `results/confusion_matrix.png`, `results/gradcam_samples.png`, `results/training_curves.png`.
- The notebook's printed outputs are the Kaggle run's stdout log and saved figures, attached after download.
- Caveat: PlantVillage images are lab photos on plain backgrounds, and a random split can put photos of the same leaf in both train and test, so expect lower accuracy on field images.

**Superseded run:** `02_train_model.ipynb` is the original CPU run, kept for the record. It reached only 29.7% val accuracy because it rescaled inputs to [0, 1] instead of using ResNet50's `preprocess_input`, augmented the validation set, and never unfroze the base. `04_kaggle_finetune.ipynb` fixes all three and is the source of every number above.

## Quickstart (Windows, cmd.exe)
Tested on Windows 11, Python 3.12, CPU only. Python 3.10 to 3.13 should work. On macOS/Linux, activate with `source .venv/bin/activate` instead.

On Windows, clone into a short path such as `C:\src`. TensorFlow's wheel contains files 163 characters deep, and pip fails with `No such file or directory` once the full path passes 260 characters, unless Windows long paths are enabled.

```cmd
git clone https://github.com/SuppaTessta/phytosentinel-crop-disease-classifier.git
cd phytosentinel-crop-disease-classifier
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```

### Get the model weights
The trained model (`phytosentinel_resnet50.keras`, 336 MB) is not in git. The demo loads it from `models/`; class names come from `results/class_names.json`. Retraining needs a Kaggle API token in `%USERPROFILE%\.kaggle\kaggle.json`.

- **Retrain it yourself** on a free Kaggle GPU (about 65 min; GPU kernels need a phone-verified Kaggle account). Set your username in `kaggle/kernel-metadata.json`, then:
  ```cmd
  kaggle kernels push -p kaggle
  kaggle kernels status YOUR_KAGGLE_USERNAME/phytosentinel-resnet50-finetune
  kaggle kernels output YOUR_KAGGLE_USERNAME/phytosentinel-resnet50-finetune -p models
  ```
  Run the last command once the status shows complete.

### Run the demo
```cmd
python 03_demo.py
```
It opens http://127.0.0.1:7860 in your browser (set `GRADIO_SERVER_PORT` to use another port). Upload a leaf image to see the image, the Grad-CAM heatmap and the overlay titled with the predicted class and confidence. On a laptop CPU the model loads in about 20 s; the first prediction takes about 8 s and later ones about 3 s.

## Notebooks
| Notebook | What it does |
|---|---|
| `01_eda_crop_disease.ipynb` | Class counts (54,305 images, 38 classes, 36x max/min imbalance) and image sizes (a 100-image sample is all 256x256). |
| `02_train_model.ipynb` | Superseded CPU run (29.7% val accuracy), see above. |
| `notebooks/04_kaggle_finetune.ipynb` | Current training and test evaluation on a Kaggle T4. `kaggle/` holds the same notebook without outputs plus the kernel metadata used by `kaggle kernels push`. |

Notebooks 01 and 02 read the dataset from `data/plantvillage/plantvillage dataset/color`. To run them locally:
```cmd
kaggle datasets download abdallahalidev/plantvillage-dataset -p data/plantvillage --unzip
```

## Layout
- `03_demo.py`: Gradio demo (prediction + Grad-CAM)
- `results/`: test metrics, class names and figures from the Kaggle run
- `kaggle/`: Kaggle kernel source and metadata
- `requirements.txt`: dependencies for the demo and the local notebooks

## License
Code: MIT (see `LICENSE`). The PlantVillage dataset (Kaggle `abdallahalidev/plantvillage-dataset`) is CC BY-NC-SA 4.0; check its terms before any commercial use of the data or of weights trained on it.
