###########
#Libraries#
###########
import cv2
import os
import pandas as pd
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.ticker as mtick

#################################################################################################################################################################

##################
#Global variables#
##################
#Resolution of the images from the resolution used during CT-scanning
resolution = 27
#Path where the main python file is
path = os.path.dirname(os.path.realpath(__file__))
#Path to the directory containing the images
image_directory = os.path.join(path,"figures")

#################################################################################################################################################################


###########
#Functions#
###########
#prepInputs: Prepares the images to process
####requires: Original image
####returns: Original Greyscaled image in array format, greyscaled mark of the original image with external contour in array format

def prepInputs(img_path):
    #Loads image into matrix format
    image = cv2.imread(img_path)
    #Grayscale image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #Apply otsu threshold to the blurred image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    #Creates final mask in order to find the contours
    mask = cv2.bitwise_and(gray, gray, mask=thresh)
    mask[thresh==0] = 255
    mask[thresh==255] = 0
    #Creates a copy of mask where we are going to store all porosities
    all_porosities = mask.copy()
    #Computes external contour of the mask image
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    #Draws the external contour in the mask image
    cv2.drawContours(mask, contours, -1, (255, 255, 255), cv2.FILLED)
    #Applies greayscale to the mask
    contouredmask = mask #= cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    #Stores all porosities in black
    all_porosities[contouredmask==0] = 255
    #Returns original greyscale image, greyscaled mask and all porosities images
    return gray,contouredmask,all_porosities

#pixelsCount: Computes the size of the image and counts different types of pixels
####requires: Grayscale original image and greyscale contoured mask
####returns: Image size, countour size, number or bright pixels and number of dark pixels
def pixelsCount(gray,contouredmask):
    #Defines image size
    imagesize = gray.shape[0]*gray.shape[1]
    masksize = 0
    #Initializes pixel counters
    darkPixels = 0
    brightPixels = 0
    #Defines threshold to sort dark from bright pixels (if greyscale is over 50, it is a bright pixel)
    darkPixelsThreshold = 50
    #Loops through the image and counts only the pixels which are inside the external contour
    for row in range(0, gray.shape[0]):
        for col in range(0, gray.shape[1]):
            #Condition ensures only pixels inside the contours are considered
            if contouredmask[row,col]==255:
                masksize += 1
                if gray[row,col]>=darkPixelsThreshold:
                    brightPixels += 1
                else:
                    darkPixels += 1
    return imagesize,masksize,brightPixels,darkPixels

#computePorosity: Computes the porosith from an image
####requires: Grayscale original image and greyscale contoured mask
####returns: Porosity of an image
def computePorosity(gray,contouredmask):
    #Calls pixelsCount function to get the number of dark pixels and the size of the countoured areas
    imagesize,masksize,brightPixels,darkPixels = pixelsCount(gray,contouredmask)
    #If no contour is detected, then porosity is 0 (Image is defective)
    #Else porosity is number of dark pixels divided by size of the contoured areas
    if masksize == 0:
        porosity = 0
    else:
        porosity = darkPixels/masksize
    #Returns porosity value
    return porosity

#approximateContourWithCircle: approximates the contour of a porosity with a circle
####requires: Coordinates of the porosity contou
####returns: Center and radius of the circle
def approximateContourWithCircle(contour):
    #Computes center of the circle
    center = np.mean(contour,axis=0)
    #Computes radius of the circle
    radius = np.max(np.linalg.norm(contour-center,axis=1))
    #Returns center and radius of the circle
    return center,radius

#approximateContourWithEllipse: approximates the contour of a porosity with an ellipse
####requires: Coordinates of the porosity contou
####returns: Center,radiuses and angle of the ellipse
def approximateContourWithEllipse(contour):
    #Computes covariance of the contour matrix
    cov_matrix = np.cov(contour.T)
    #Computes eigenvalues and eigenvectors of covariance matrix
    evals, evecs = np.linalg.eig(cov_matrix)
    #Finds index of largest eigenvalue
    major_axis = np.argmax(evals)
    #Computes center of the ellipse
    center = np.mean(contour, axis=0)[::-1]
    #Computes radiuses of the ellipse
    radius = np.sqrt(5.991 * evals)
    #Computes angle of the ellipse
    angle = np.degrees(np.arctan2(evecs[major_axis, 1], evecs[major_axis, 0]))
    #Returns center, rediuses and angle of the ellipse
    return center,radius,angle

#################################################################################################################################################################

##################
#Debug Procedures#
##################
#exportDebugMatrix: Exports the image into a csv file (matrix format)
####requires: Greyscale image and a name to the csv to create
####returns: A csv file of the image containing grayscale value of each pixel

def exportDebugMatrix(image,filename):
    DF = pd.DataFrame(image)
    DF.to_csv(os.path.join(path,filename),sep=',',index=False,header=False)

#showAllPorosities: Draws the contours of all porosities on the  greyscale image
####requires: Image name (string format)
####returns: Image with plotted contours around all identified porosities
def showAllPorosities(imagename):
    gray,contouredmask,all_porosities = prepInputs(os.path.join(image_directory,imagename))
    porosities_contours = measure.find_contours(all_porosities, 0)
    fig, ax = plt.subplots()
    ax.imshow(gray, cmap=plt.cm.gray)
    for contour in porosities_contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

#drawCircleEllipseContours: Approximates the contours of all porosities with a circle or an ellipse on the greyscale image
####requires: Image name (string format)
####returns: Image with drawn circles and ellipses which approximate best existing porosities
def drawCircleEllipseContours(imagename):
    gray,contouredmask,all_porosities = prepInputs(os.path.join(image_directory,imagename))
    porosities_contours = measure.find_contours(all_porosities, 0)
    fig, ax = plt.subplots()
    ax.imshow(gray,cmap=plt.cm.gray)
    for contour_idx,contour in enumerate(porosities_contours):
        y_range = np.max(contour[:,0]) - np.min(contour[:,0])
        x_range = np.max(contour[:,1]) - np.min(contour[:,1])
        if x_range != 0:
            aspect_ratio = y_range / x_range
        else:
            aspect_ratio = 2
        if (aspect_ratio < 1.5) & (aspect_ratio > 0.5):
            center,radius = approximateContourWithCircle(contour) 
            ax.add_artist(plt.Circle(center[::-1], radius, color='r', fill=False))
        else:
            center,radius,angle = approximateContourWithEllipse(contour)
            if aspect_ratio < 1:
                ax.add_artist(Ellipse(center, radius[0]*2, radius[1]*2, angle=angle, fill=False, edgecolor='b'))
            else:
                ax.add_artist(Ellipse(center, radius[1]*2, radius[0]*2, angle=angle, fill=False, edgecolor='b'))
    plt.show()

#################################################################################################################################################################

############
#Procedures#
############
#exportPorosityMetricsData(): Creates a csv file that computes the porosity metric for each image in the directory
####requires: Collected porosity percentage from every image
####returns: CSV with image name and porosity percentage
def exportPorosityMetricsData(resolution,FirstPic,LastPic):
    porositymetric = pd.DataFrame(columns=['Image','Z-Height','Porosity'])
    for f in os.listdir(image_directory):
        if f[-4:] == ".png":
            gray,contouredmask,all_porosities = prepInputs(os.path.join(image_directory,f))
            porosity = computePorosity(gray,contouredmask)
            porositymetric = pd.concat([porositymetric,pd.Series([f,int(f[-8:-4])*resolution,porosity],index=porositymetric.columns).to_frame().T],ignore_index=True)
    porositymetric[FirstPic:LastPic+1].to_csv(os.path.join(path,'1_porosity_metrics.csv'),index=False)
    return porositymetric[FirstPic:LastPic+1]


#exportPorosityDistributionData(): Creates a csv file that computes the distribution of porosities for each image in the directory
####requires: Collected porosity distribution data from every image
####returns: CSV with image name and corresponding porosities detected, IDed and geometrical features determined
def exportPorosityDistributionData(resolution,FirstPic,LastPic):
    shapes = pd.DataFrame(columns=['Image','PorosityId','ShapeType','Center_x','Center_y','Z-Height','Major_Radius','Minor_Radius','Angle'])
    for f in os.listdir(image_directory):
        if f[-4:] == ".png":
            gray,contouredmask,all_porosities = prepInputs(os.path.join(image_directory,f))
            porosities_contours = measure.find_contours(all_porosities, 0)
            for contour_idx,contour in enumerate(porosities_contours):
                y_range = np.max(contour[:,0]) - np.min(contour[:,0])
                x_range = np.max(contour[:,1]) - np.min(contour[:,1])
                if x_range != 0:
                    aspect_ratio = y_range / x_range
                else:
                    aspect_ratio = 2
                if (aspect_ratio < 1.5) & (aspect_ratio > 0.5):
                    center,radius = approximateContourWithCircle(contour) 
                    shapes = pd.concat([shapes,pd.Series([f,contour_idx,'Circle',center[0],center[1],int(f[-8:-4])*resolution,radius,radius,0],index=shapes.columns).to_frame().T],ignore_index=True)
                else:
                    center,radius,angle = approximateContourWithEllipse(contour)
                    if aspect_ratio < 1:
                        shapes = pd.concat([shapes,pd.Series([f,contour_idx,'Ellipse',center[0],center[1],int(f[-8:-4])*resolution,radius[1]*2,radius[0]*2,angle],index=shapes.columns).to_frame().T],ignore_index=True)
                    else:
                        shapes = pd.concat([shapes,pd.Series([f,contour_idx,'Ellipse',center[0],center[1],int(f[-8:-4])*resolution,radius[0]*2,radius[1]*2,angle],index=shapes.columns).to_frame().T],ignore_index=True)
    shapes[(shapes['Image'].str[-8:-4].astype(int) >= FirstPic) & (shapes['Image'].str[-8:-4].astype(int) <= LastPic)].to_csv(os.path.join(path,'1_porosity_distribution.csv'),index=False)
    return shapes[(shapes['Image'].str[-8:-4].astype(int) >= FirstPic) & (shapes['Image'].str[-8:-4].astype(int) <= LastPic)]

#################################################################################################################################################################

#############
#Main Script#
#############

def percentage_formatter(x,pos):
    return '{:.0%}'.format(x)

if __name__ == '__main__':
  #Write down the starting image number and last one
    FirstPic = 10
    LastPic = 166
    porositymetric = exportPorosityMetricsData(resolution,FirstPic,LastPic)
    plt.figure()
    plt.hist(porositymetric['Porosity'],bins=10,edgecolor='black')
    plt.gca().xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x*100:.3}%'))
    plt.gca().yaxis.set_major_locator(mtick.MultipleLocator(10))
    plt.xlabel('Porosity [%]')
    plt.ylabel('Number of Porosity Instances')
    plt.title('Instances of Porosity')
    plt.figure()
    plt.scatter(porositymetric['Z-Height'],porositymetric['Porosity'])
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x*100:.3}%'))
    plt.xlabel('Height [um]')
    plt.ylabel('Porosity [%]')
    plt.title('Distribution of Porosity along the Small Cube Height')
    plt.figure()
    plt.plot(porositymetric['Z-Height'],porositymetric['Porosity'])
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x*100:.3}%'))
    plt.xlabel('Height [um]')
    plt.ylabel('Porosity')
    plt.show()


    porodist = exportPorosityDistributionData(resolution,FirstPic,LastPic)
    plt.figure()
    shape_colors = {'Ellipse':'blue','Circle':'red'}
    shape_markers = {'Ellipse':'s','Circle':'o'}
    for shape in shape_colors:
        mask = porodist['ShapeType'] == shape
        plt.scatter(porodist['Center_x'][mask],porodist['Center_y'][mask],color=shape_colors[shape],marker=shape_markers[shape],label=shape,s=10)
    plt.xlabel('X Position [px]')
    plt.ylabel('Y Position [px]')
    plt.legend(loc='upper right')
    plt.title('Distribution of Porosity Shapes along the Small Cube CS')
    shapecount = porodist.groupby(['Image','Z-Height'])['ShapeType'].value_counts().unstack().fillna(0).reset_index()
    plt.figure()
    plt.plot(shapecount['Z-Height'],shapecount['Circle'],'r',label='Circle')
    plt.plot(shapecount['Z-Height'],shapecount['Ellipse'],'blue',label='Ellipse')
    plt.xlabel('Height [um]')
    plt.legend()
    plt.show()



