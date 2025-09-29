# Brain Tumor Classification using VGG16

A deep learning-based web application for classifying brain MRI scans into different tumor categories using transfer learning with VGG16 architecture.

## Project Overview

This project implements a brain tumor classification system that can distinguish between:
- **Glioma Tumor**
- **Meningioma Tumor** 
- **Pituitary Tumor**
- **No Tumor**
- **Unknown** (external objects)

The model uses VGG16 with transfer learning and is deployed as a user-friendly web application using Streamlit.

## Project Structure

```
brain-tumor-classification/
├── app.py                 # Streamlit web application
├── model.py              # Model training script
├── my_model.h5           # Pre-trained model (download separately)
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── braintumors/         # Dataset directory
    ├── train/
    │   ├── glioma_tumour/
    │   ├── meningioma_tumour/
    │   ├── no_tumour/
    │   └── pituitary_tumour/
    ├── validation/
    │   ├── glioma_tumour/
    │   ├── meningioma_tumour/
    │   ├── no_tumour/
    │   └── pituitary_tumour/
    └── test/
        ├── glioma_tumour/
        ├── meningioma_tumour/
        ├── no_tumour/
        └── pituitary_tumour/
```

## Model Details

### Architecture
- **Base Model**: VGG16 (pre-trained on ImageNet)
- **Custom Layers**:
  - Flatten layer
  - Dense layer (512 units, ReLU activation)
  - Dropout layer (0.5 rate)
  - Output layer (5 units, Softmax activation)

### Training Parameters
- **Image Size**: 224×224 pixels
- **Batch Size**: 64
- **Epochs**: 15
- **Optimizer**: Adam (learning rate: 0.0001)
- **Loss Function**: Categorical Crossentropy

### Data Augmentation
- Rotation range: 30°
- Width/height shift: 20%
- Shear range: 20%
- Zoom range: 20%
- Horizontal flip: True

## Installation & Setup

### Prerequisites
- Python 3.7+
- TensorFlow 2.x
- Streamlit

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd brain-tumor-classification
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the pre-trained model**
   Since the model file is large (~500MB), download it from the link below and place it in the project root directory:

   [**Download my_model.h5**](https://drive.google.com/your-model-link-here) *(Replace with actual link)*

   Alternatively, you can train the model yourself (see Training section).

4. **Organize your dataset**
   Ensure your dataset follows the structure shown above in the `braintumors` directory.

## Usage

### Running the Web Application

1. **Start the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and go to `http://localhost:8501`

3. **Upload a brain MRI image** (JPG, JPEG, or PNG format)

4. **Click "Classify Image"** to get the prediction

### Training the Model

If you want to train the model from scratch:

1. **Update the dataset paths** in `model.py`:
   ```python
   TRAIN_DIR = "path/to/your/braintumors/train"
   TEST_DIR = "path/to/your/braintumors/validation"
   TEST_DIR_FINAL = "path/to/your/braintumors/test"
   ```

2. **Run the training script**:
   ```bash
   python model.py
   ```

## Results

The model typically achieves:
- **Training Accuracy**: ~95%
- **Validation Accuracy**: ~85-90%
- **Test Accuracy**: ~85%

## Dependencies

Create a `requirements.txt` file with the following:

```txt
tensorflow==2.13.0
streamlit==1.28.0
pillow==10.0.1
numpy==1.24.3
matplotlib==3.7.2
```

Install all dependencies using:
```bash
pip install -r requirements.txt
```

## File Descriptions

- **app.py**: Streamlit web application for image classification
- **model.py**: Script for training the VGG16-based model
- **my_model.h5**: Pre-trained model weights (download separately)

## Dataset

The model is trained on a brain tumor MRI dataset containing four classes. Ensure your dataset is properly organized with separate directories for training, validation, and testing.

## Limitations

- The model is trained on specific MRI scan types and may not generalize well to all medical imaging formats
- Clinical use requires further validation and should not replace professional medical diagnosis
- Performance may vary with image quality and scanning parameters

## Future Improvements

- Integration of more advanced architectures (ResNet, EfficientNet)
- Confidence thresholding for medical safety
- Multi-modal imaging support
- Real-time classification capabilities

## License

This project is for educational and research purposes. Always consult healthcare professionals for medical diagnoses.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for suggestions and bug reports.

---

**Note**: This tool is intended for educational and research purposes only and should not be used for actual medical diagnosis without proper clinical validation.

