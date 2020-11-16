# Objects-Detection-in-Images-using-YOLOv2
<br /><br />
## Introduction

Useful of knowledge before start:
+ YOLO 2016 Paper [YOLO: Unified, Real-Time Object Detection](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf) and [This website](https://pjreddie.com/darknet/yolo/) would be good guides to understand the exact concept.
+  The new version of YOLO was a paper in 2017 that was named [YOLO9000: Better, Faster, Stronger](https://openaccess.thecvf.com/content_cvpr_2017/papers/Redmon_YOLO9000_Better_Faster_CVPR_2017_paper.pdf).
+ The interesting and updates version of the YOLO topic is [YOLOv3 2018: An Incremental Improvement](https://arxiv.org/pdf/1804.02767.pdf).

----------------------------------------------------------------------------------------------------
<br />
The main goal of this challenge is to recognize objects from several visual object classes in realistic scenes. It is fundamentally asupervised learning  problem in  that a training set of labelled images is provided.
In recent years, Object Detection is one of the fundamental challenges in the field of computer vision,image processing and pattern recognition, which provides aresearch foundation and a theoretical basis for a series ofautomatic analysis and understanding of image content suchas target tracking and behaviour analysis. Object detection has been a challenging task for decadessince images of objects in the real world environment areaffected by illumination, rotation, scale, and occlusion. In recent years, a large improvement in imagerecognition was made by a series of CNN based solutions. This is a technology that mixes Artificial Neural Networks (ANN) and up to date deep learning strategies. CNN is designed to automatically and adaptively learn spatial hierarchies of features through backpropagation by using multiple building blocks, such as convolution layers,pooling layers, and fully connected layers. <br />

This project exhibits one of the best frame obstaclesdetection based on Convolutional Neural Networks (CNN) Meth-ods known as ”You Only Look Once” (YOLO)v2. This  methodefficiently classifies moving or stationary objects as well as bytype. In this perspective, two values of anchor boxes (fourand eight with different sizes) are selected for comparing theloss. Also, the effects of loss weights and optimizer methods areexplored. This model is trained and tested based on the PASCALVOC2012 dataset. The comparing results show that increasingthe anchor boxes and also selecting Nadam optimizer improvesthe training performance while changing the loss weights does not have a positive effect on the accuracy. <br />

This deep network is better than other alternative deep learning networks in applications such as computer vision and Natural Language Processing (NLP) asit can mitigate the error rate significantly and hence improvenetwork performances. YOLO is refreshingly simple (see Figure 1), a single convolutional network simultaneously predictsmultiple bounding boxes and class probabilities for thoseboxes. YOLO trains on full images and directly optimizesdetection performance. This unified model has several benefitsover traditional methods of object detection. <br />


<p align="center">
<img width="650" height="200" alt="d1" src="https://user-images.githubusercontent.com/71558720/99200627-676fc580-2774-11eb-9e89-5092efc722a8.png">
</p>
<p align="center">
<em>Fig.1: The YOLO Detection System. Processing images with YOLO is simple and straightforward. The system (1) resizes the input image to 448×448, (2) runs a single convolutional network on the image, and (3) thresholds the resulting detections by the model’s confidence</em>
</p> <br /> 

#### Problem formulation:  
The main scientific questions of thispaper are as follow: <br /> 

+ To explore how the loss value of the YOLO v2 changes byintroducing a new anchor box set in the design method.
+ Changing the training optimizer factor to improve the network performance.
+ To test the effect of different components of the loss function (by changing their weights) on the overall accuracyof the network. <br /> <br /> 

## General Method:
The YOLO detection model has incomparable advantage inexecution speed. However, the detection accuracy is slightlyless than other approaches especially in the detection of smalltargets. To solve these problems, Redmonet.al. proposed anupgraded version of the YOLO detection model known as YOLOv2 detection model. This upgraded model not only ensures the absolute superiority of the YOLO detection model in detection speed but also greatly improves the detection accuracy of the model by introducing some optimization methods. Firstly, some of the techniques used in YOLOv2 detection models arebatch normalization, high resolution, fine-grained features classifier, convolutional with anchor boxes, dimension clusters, direct location prediction, multi-scale trainingand so on. Secondly, the structure of CNN network model which YOLO detection model relies on is adjusted. <br />


## Network Architecture:
The network model contains 22 convolution layers and 5 max-pooling layers. YOLO algorithm divides any given inputimage into a S×S grid system. Each grid on the input image is  responsible for detection on an object and each grid cell predicts B bounding boxes together (width, height, box center x and y) with their confidence score. Each confidence  scorereflects the probability of the predicted box containing anobject Pr(Object), as well as how accurate is the predicted box by evaluating its overlap with the ground truth bounding box measured by intersection over union IoU. A boundingbox describes the rectangle that encloses an object. YOLO alsooutputs a confidence score that tells us how certain it is that the predicted bounding box actually encloses some object. The architecture contains repeatedly, stacks Convolution + Batch Normalization + Leaky Relu + Maxpooling2D layers. <br /> 

Following  YOLO, the objectness prediction still predicts the IoU of the ground truth and the proposed box and the class predictions predict the conditional probability of  that  classgiven that there is an object. Also in this architecture, the network is shrunk to operate on 416 input images instead of 448×448. The reason is an odd number of locations  in the feature map are needed so there is a single-center cell. YOLO’s convolutional layers downsample the image by a factor of 32 so by using an input image of 416 we get an output resolutionfeature map of 13×13. In our analysis, we take 8 anchorboxes, hence the output dimension of the last layer would be 13×13×200. <br /> 

<p align="center">
<img width="900" height="500" alt="net new" src="https://user-images.githubusercontent.com/71558720/99200623-663e9880-2774-11eb-9e64-6880c56e5114.PNG">
</p>
<p align="center">
<em>Fig.2: YOLOv2 Architecture</em>
</p> <br /> 



## Special improvements in YOLOv2:
#### 1) Dimension clusters: 
With YOLO the box dimensions are handpicked YOLOv2 uses k-means clustering to conduct clusteringanalysis on the size of the object bounding boxes. <br />
#### 2) Intersection of Union (IoU): 
This is calculated by dividing the overlapped area of a predicted box and the truth box by the whole area made. <br /> 

For knowing how many anchor boxes should be used, asolution is to use the plot of mean IoU vs K clusters. This shows the true number of clusters captured when the increase in  the mean IoU slope is ”substantially” large (see Figure 3).

<p align="center">
<img width="300" height="140" alt="11" src="https://user-images.githubusercontent.com/71558720/99202903-c4707900-277e-11eb-8521-1b3656a86f0e.PNG">
</p> <br /> 

<p align="center">
<img width="650" height="400" alt="dd" src="https://user-images.githubusercontent.com/71558720/99200628-68085c00-2774-11eb-965b-f87aa4dcb94e.png">
</p>
<p align="center">
<em>Fig.3: The clustered anchor box information.</em>
</p> <br /> 


Here, we want to explore the effect of 4 amd 8 anchor boxes parameters on the final result. In this case the clusters are shown in Figure 4.

<p align="center">
  <img width="350" height="290" hspace="20" alt="k4" src="https://user-images.githubusercontent.com/71558720/99200620-650d6b80-2774-11eb-8a62-df51e991704d.png"> <em>K=4</em>
  <img width="350" height="290" hspace="20" alt="k8" src="https://user-images.githubusercontent.com/71558720/99200621-65a60200-2774-11eb-9c74-f18dbdc95e26.png"> <em>K=8</em>
</p> <br /> 
<p align="center">
<em>Fig.4: Visualization of bounding boxes clusters for K=4, K=8.</em>
</p> <br /> <br /> 

| Anchor box           | Width   | Height |
| --------------------|:---:|:---:|
| K=1  | 1.07709 | 1.78171| 
| K=2  | 2.71054 | 5.12469 |
| K=3  | 10.47181 | 10.09646  |
| K=4    |5.48531| 8.11011  | 
 
 <em>TABLE I: The shape of four anchor boxes</em> <br /><br />

| Anchor box           | Width   | Height |
| --------------------|:---:|:---:|
| K=1  |0.80658  | 1.31287| 
| K=2  | 2.32824   | 6.54707 |
| K=3  | 11.32239  | 11.10198 |
| K=4    |9.05663 | 5.76735 | 
| K=5    |6.90248 | 10.36930  | 
| K=6   |4.40284  | 3.77522  | 
| K=7    | 1.72716  | 3.04532| 
| K=8    |4.21712 | 8.77177| 

 <em>TABLE II: The shape of eight anchor boxes</em> <br />
 
  
#### 3) Input/Output Encoding: 
Data set contains many imageswith different sizes. Input/output encoding is necessary to reshape any image to the pre-specifed shape. The input encodingessentially only requires to readin and resize the image to thepre-specified shape. The output’s xmin, ymin, xmax and ymax also need to be resized and assign each object to a groundtruth anchor box. Also, the bounding boxes encoding for out-put is done by defining (centerx,centery,centerw,centerh) formats as shown in Figure 5. <br />

Centerx=1/2(xmin+xmax)
Centery=1/2(ymin+ymax)
Centerw=  (xmax−xmin)
Centerh=  (ymax−ymin) <br />


<p align="center">
<img width="350" height="300" alt="encode" src="https://user-images.githubusercontent.com/71558720/99200630-69398900-2774-11eb-9571-0a90744a4858.png">
</p>
<p align="center">
<em>Fig.5: Bounding box encoding.</em>
</p> <br /> 

#### 4) Direct Location Prediction:

Using anchor boxes in YOLO causes a model instability, especially during early iteration. That mostly comes from predicting the (x,y) location forthe box. The network predicts 5 bounding boxes at each cell in the output feature map. The network predicts 5 coordinates for each bounding box, tx,ty,tw,th. If the cell is offset from the top left corner of the image by(cx,cy) and the bounding box prior has width and height pw,ph, then the predictions correspond to:

<p align="center">
<img width="300" height="140" alt="22" src="https://user-images.githubusercontent.com/71558720/99202901-c4707900-277e-11eb-9474-b4ba3343a0b3.PNG">
</p> <br /> 


#### 5) Fine-Grained Features:
YOLOv2 predicts detection on a 13×13 feature map, which does a sufficient job at detecting large objects, but it comes short when detecting small-scaleobjects as we get deeper  in the network. That being said, using higher-resolution feature maps helps the network detect objects of different scales, so YOLOv2 adapts this approach, but instead of stacking a layer of high-resolution on top of the convolution layers, the features of a 26×26 resolution layer are concatenated  with the low resolution features along  thechannels, making the 26×26×512 feature map a 13×13×2048 feature map, similar to identity mappings in ResNet. <br /> 



#### 6) Multi-Scale Training:

Since the network consists only of convolutional and pooling layers, not fully connected layers, it can be trained on different input sizes, thus detecting well on different  resolutions. Therefore, during training for every 10 batches, the network randomly chooses a new image size, since the model downsamples by a factor of 32, the chosen sizes should be a multiple of 32. <br /> <br /> 


## Loss Function
The loss function of YOLOv2 is quite complex. In this version the fully connected layers is removed from YOLO(v1) and use anchor boxes to predict bounding boxes. Overall Loss is the summation of Loss calculated for bounding boxes, classes and confidence. The loss corresponding to (grid cell,anchor box)pair =(i,j) is calculated as following equations: <br /> <br />

<p align="center">
<img width="500" height="1000" alt="33" src="https://user-images.githubusercontent.com/71558720/99203647-29c56980-2781-11eb-8b89-2d6115c6ad27.PNG">
</p> <br /><br />


## Dataset
In the analysis, the PASCAL VOC 2012 data set have beenused. The data set size is 2,0GB and is a well-known data setfor object detection, classification, segmentation of objects. There are around 10,000 images for training and validation containing bounding boxes with objects. It has been split into 50% for training/validation and 50% for  testing. The main goal of this challenge is to recognize objects from several visual object classes in realistic scenes. It is fundamentally asupervised learning  problem in  that a training set of labelled images is provided. The total number of objects is 40138 in 17125 images fitting in 20 classes. Object classes are definedas follows:
<br />

```diff
+ Person:   person
+ Animal:   bird, cat, cow, dog, horse, sheep
+ Vehicle:  aeroplane,  bicycle,  boat,  bus,  car,  motorbike,train
+ Indoor:   bottle,  chair,  dining  table,  potted  plant,  sofa,tv/monitor
```
<br />

------------------------------------------------------------------------------
# EXPERIMENTAL RESULT 
<br /><br />

<p align="center">
<img width="500" height="330" alt="44" src="https://user-images.githubusercontent.com/71558720/99204135-b02e7b00-2782-11eb-9c00-bfe0be413091.PNG">
</p>
<p align="center">
<em>TABLE III: System setup</em>
</p> <br />


## Result and Analysis:

At first, the effect of increasing the anchor boxes is illustrated in Figure-6. It is clear that increasing the number of anchor boxes increases the accuracy by decreasing the loss function for similar epochs.

<p align="center">
<img width="700" height="350" alt="plott" src="https://user-images.githubusercontent.com/71558720/99200625-66d72f00-2774-11eb-89cb-7bbaf04a36c6.PNG">
</p>
<p align="center">
<em>Fig.6: Loss function comparison for two anchor boxes sets K=4, K=8</em>
</p> <br />


We are also interested to see the weight of each loss components in detection accuracy. This is another scientifictest performed through changing the loss functions weight. To  do that, weights for bounding box parameters loss and confidence loss are doubled. After changing weights (TableIV), the network is trained for 50 epochs and the comparison results are shown in Figure-7. The comparison shows that increasing loss components weights does not increase the accuracy although it was expected to get better results  sinceloss function on the detection part has more sensitivity on the localization errors. 


<p align="center">
<img width="700" height="350" alt="lambda" src="https://user-images.githubusercontent.com/71558720/99200622-663e9880-2774-11eb-9022-cf055e5ab981.PNG">
</p>
<p align="center">
<em>Fig.7: Effect of loss weights on the training accuracy on 50 epochs</em>
</p> <br /><br />



| ----       | λ coord   | λ class | λ object |λ no−object|
| --------------------|:---:|:---:|:---:|:---:|
| Initial lambda | 1.0 | 1.0 | 5.0 | 1.0| 
| New lambda | 2.0 | 1.0 | 10.0 | 1.0|

<em>TABLE IV: Loss weight values for both runs</em> <br /><br />


Finally, the effects of optimizer methods on the training results are explored. In the original work, the network wastrained with Initial Adam optimizer. Table V shows the optimizer tested methods with corresponding values and Figure 8 illustrates that using Nadam provides the best performance by reducing the cost in shorter time.
<br />


|Optimizer | Parameters  used  |
| --------------------|:---:|
| Initial Adam | lr=0.5e-4, beta-1=0.9, beta-2= 0.999, epsilon=1e-07|
| New Adam |lr=0.0001, beta-1=0.9, beta-2= 0.999, epsilon=1e-05| 
| Nadam |lr=0.001, beta-1=0.9, beta-2= 0.999, epsilon=1e-07|
|Adamax |lr=0.001, beta-1=0.9, beta-2= 0.999, epsilon=1e-07|
|SGD |lr=1e-04,  decay=0.0005, momentum=0.9|

<br /><br />


<p align="center">
<img width="700" height="350" alt="opt" src="https://user-images.githubusercontent.com/71558720/99200624-66d72f00-2774-11eb-8826-8c57ba48847e.PNG">
</p>
<p align="center">
<em>Fig.8: Effects of optimizers on loss values</em>
</p> <br /><br /><br />





## ** Mina R **
