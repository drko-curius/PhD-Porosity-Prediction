# About the Porosity Prediction Repository

As part of my doctoral research, this repository contains the scripts used for porosity analysis and prediction of CT-scanned 3D-printed cubic specimens.  

It is laid out as follows:

PhD-Porosity-Prediction/  
                                                                                                                                                                                  
├── Porosity-Analysis/                  Code, example of input and output for porosity analysis  
                                                                                                                                                                                  
│├── Script/                          Python code for porosity analysis, used to extract different porosity metrics from .png exploitable images of CT-scanned cubes  
│├── Example_Data/                    Sample image for testing the code, and example of the script outputs  
││├────── Input_Image/                Example of image for testing the code.  For the full dataset, refer to the About_Data file in the repository  
││├────── Output_CSV/                 Example of the csv files generated using the script. Note the provided code runs on an image folder  
││├────── Output_Plots/               Example of the plots generated using the script. Colored contours are qualitative plots. Rest is quantitative   
                                                                                                                                                                                  
                                                                                                                                                                                  
├── Machine-Learning/                   The ML scripts developed for classification and prediction  
                                                                                                                                                                                  
│├── CNN/                             Convolutional Neural Network model architecture and example of outcome    
││├────── Example/                  Example of sorting between defective and exploitable images  
│││├───────── Defective/               Example of images the CNN is accurately classifying as "defective"  
│││├───────── Exploitable/             Example of images the CNN is accurately classifying as "exploitable"  
│││├───────── Unsorted/                Link to an example of unsorted folder of images (All folders from the Zenodo dataset are unsorted)  
││├────── Output/                   Final accuracy and loss plots  
││├────── Script/                   CNN model architecture and use  
│├── MLP/                            Multi-Layer Perceptron model architecture and example of dataset  
││├────── Example_of_Dataset/         Example of dataset for neural network training  
││├────── Output/                     Final MSE and R squared plots for the example dataset  
││├────── Script/                     MLP model architecture and use  
                                                                          
                                                                                                                                                                                  
├── About_Data                         : Link to the full dataset  
                                                                                                                                                                                  
├── README.md                         :  This document summarizing the content of the repository  
                                                                                                                                                                                  
├── LICENSE                         :    MIT license. Please credit for use  
                                                                                                                                                                                  
