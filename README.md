# PhD-Porosity-Prediction

As part of my doctoral research, the following code has been used for the porosity analysis and prediction in CT-scanned 3D printed cubic specimens.

Porosity Analysis Script : Overall script used to extract porosity metrics from the cross-sections of the CT-scanned specimens.

CNN Image Classifier : Not all images obtained from the CT-scanned specimens are exploitable. Therefore, a CNN was developed to sort the images in two categories: Defective and exploitable. Only exploitable images would yield results in porosity analysis

MLP Porosity Predictor : a MLP is trained to predict porosity percentage at specific layer heights from the process parameters
