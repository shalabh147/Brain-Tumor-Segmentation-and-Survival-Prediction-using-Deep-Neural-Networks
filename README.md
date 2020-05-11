# Brain-Tumor-Segmentation-and-Survival-Prediction-using-Deep-Neural-Networks

Keras implementation

###Dataset
For accessing the BRATS dataset upto 2016, you need to create account with https://www.smir.ch. While for Brats dataset after 2016, create an accound with https://www.med.upenn.edu/sbia/brats2018/registration.html and apply for the dataset.

## Using the code
You are free to use contents of this repo for academic and non-commercial purposes only.

## Resources
As such, this code is not an implementation of a particular paper,and is combined of many architectures and deep learning techniques from various research papers on Brain Tumor Segmentation and survival prediction. Some of the best resources used are mentioned below.

https://arxiv.org/pdf/1505.03540.pdf : Patch based Brain Tumor Segmentation
https://www.biorxiv.org/content/10.1101/760157v1.full.pdf : Encoder Decoder network with dice loss
https://arxiv.org/pdf/1802.10508v1.pdf : Unet Architecture
https://arxiv.org/pdf/1903.11593.pdf : Survival Prediction Idea of extracting features
https://link.springer.com/chapter/10.1007/978-3-319-75238-9_17 : Integrating results along the 3 axis and different models
https://link.springer.com/chapter/10.1007/978-3-319-75238-9_30: Inception U-Net



## Key Points and Modifications
- #####  Weighted-categorical-loss function has been used to tackle with the problem of imbalanced dataset.
- ##### Both batch normalisation and dropout have been used to reduce overfitting,smoothen the optimisation curve and lead to faster convergence to global minima.


## Task
Task is of segmenting various parts of brain i.e. labeling all pixels in the multi-modal MRI images as one of the following classes:
- Necrosis
- Edema
- Non-enhancing tumor
- Enhancing tumor 
- Everything else

Brats 2015 dataset composed of labels 0,1,2,3,4 while Brats 2017 dataset consists of only 0,1,2,4.

## BRATS Dataset 
I have used BRATS 2017 training dataset for the analysis of the proposed methodology. It consists of real patient images as well as synthetic images created by MICCAI. Each of these folders are then subdivided into High Grade and Low Grade images. For each patient, four modalities(T1, T1-C, T2 and FLAIR) are provided. The fifth image has ground truth labels for each pixel. The dimensions of images are (240,240,155) in both.



## Dataset pre-processing 
Model has been trained on only those slices having all 4 labels(0,1,2,4) to tackle class imbalance and label 4 has been converted into label 3 (so that finally the one-hot encoding has size 4).

## Model Architecture 
#Patch Based Basic U-Net Architecture :
Consists of Encoder architecture,which encodes features while decreasing resolution and increasing feature maps, followed by an upsampling Decoder architecture which uses Transposed Convolution or Deconvolution to upsample the intermediate feature maps and retrieve the original resolution. 
Training on patches of size 128*128 from the second dimension (2D slices of (240,155)).

![](Captures/U-net.png)

#Full Image U-Net
Training on 2nd dimension of the images(2D slices of dimension 240*155) since 3D U-Net is computationally very expensive and hence undesirable. Limitation is unable to learn 3D neighbourhood features.

#U-Net with Inception
https://link.springer.com/chapter/10.1007/978-3-319-75238-9_30 shows inception model U-net gives better results than conventional U-Net architecture and is thus preferred. In inception model, convolutions and poolings are averaged over 1*1 , 3*3 and 5*5 sized kernels, thus taking care of high receptive fiels as well as integration of local and global features.

#Combining along the 3 views
As shown by https://link.springer.com/chapter/10.1007/978-3-319-75238-9_17, I created different models for axial,sagittal and coronal 2D views of the 3D modality and trained on each of them. After that the 3 models were combined to predict labels for each image. The combination can be done using max or average of probabilities predicted by the 3 models for the 4 classes on each pixel.Then the final prediction is made using argmax function.

![](Captures/ensembling.png)

#Survival Prediction Model
XGBoost Regression (Extreme gradient boosting regressor) has been found to perform really well in regression related tasks. It is one of the model used to train on features extracted from the bottleneck layer of the U-Net . This idea has been taken from https://arxiv.org/pdf/1903.11593.pdf where it has been shown that lung cancer survival features are somehow connected to the bottleneck layer features of the segmentation U-Net. Another network used is a feedforward neural network taking reduced features(using k-medoids clustering) from images and being trained for regression. 




## Training
### Loss function
Weighted categorical crossentropy function used as loss function in  brain tumor segmentation. One Hot encoding used and last layer activation is softmax. Trained on balanced dataset first to not let the network be overwhelmed by the dominant '0' label.
   

