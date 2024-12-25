# PhD-Porosity-Prediction

As part of my doctoral research, the following repository contains the scripts used for porosity analysis and prediction in CT-scanned 3D printed cubic specimens.
It is laid out as follows:
PhD-Porosity-Analysis/
│
├── README.md                          # This document summarizing the content of the repository
├── LICENSE                            # MIT license. Please credit for use
├── Porosity-Analysis/                 # Code, example of input and output for porosity analysis
│   ├── Script/                        # Python code for porosity analysis
│   ├── Example_Data/                  # Sample image for testing the code, and example of the script outputs
│
│
├── Machine Learning/                  # The ML scripts developped for classification and prediction
│   ├── CNN/                           # Convolutional Neural Network model architecture and example of outcome
|       ├── Script/                    # CNN model architecture and use
|       ├── Example/                   # Example of sorting between defective and exploitable images
│       ├── Output/                    # Final Accuracy and loss plots
│   ├── MLP/                           # Multi-Layer Perceptron model architecture and example of dataset
|       ├── Script/                    # MLP model architecture and use
|       ├── Example_of_Dataset/        # Example of dataset for neural network training
│       ├── Output/                    # Final Accuracy and loss plots
│
├
└── docs/                              # Documentation
    ├── setup.md                       # Instructions for setting up the environment
    └── usage.md                       # Instructions for using the tools
