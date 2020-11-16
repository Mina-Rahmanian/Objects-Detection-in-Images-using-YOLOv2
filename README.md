# Objects-Detection-in-Images-using-YOLOv2
<br /><br />
## Introduction

Useful of knowledge before start:
+ YOLO 2016 Paper [YOLO: Unified, Real-Time Object Detection](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf) and [This website](https://pjreddie.com/darknet/yolo/) would be good guides to understand the exact concept.
+  The new version of YOLO was a paper in 2017 that was named [YOLO9000: Better, Faster, Stronger](https://openaccess.thecvf.com/content_cvpr_2017/papers/Redmon_YOLO9000_Better_Faster_CVPR_2017_paper.pdf).
+ The interesting and updates version of the YOLO topic is [YOLOv3 2018: An Incremental Improvement](https://arxiv.org/pdf/1804.02767.pdf).

----------------------------------------------------------------------------------------------------
<br />
In  recent years, Object Detection is one of the fundamental challenges in the field of computer vision,image processing and pattern recognition, which provides aresearch foundation and a theoretical basis for a series ofautomatic analysis and understanding of image content suchas target tracking and behaviour analysis. Object detection has been a  challenging task for decadessince images of objects in the real world environment areaffected by illumination, rotation, scale, and occlusion. In recent years, a large improvement in imagerecognition was made by a series of CNN based solutions. This is a technology that mixes Artificial Neural Networks (ANN) and up to date deep learning strategies. CNN is designed to automatically and adaptively learn spatial hierarchies of features through backpropagation by using multiple building blocks, such as convolution layers,pooling layers, and fully connected layers. <br />

This project exhibits one of the best frame obstaclesdetection based on Convolutional Neural Networks (CNN) Meth-ods known as ”You Only Look Once” (YOLO)v2. This  methodefficiently classifies moving or stationary objects as well as bytype. In this perspective, two values of anchor boxes (fourand eight with different sizes) are selected for  comparing theloss. Also, the effects of loss weights and optimizer methods areexplored. This model is trained and tested based on the PASCALVOC2012 dataset. The comparing results  show that increasingthe anchor boxes and also selecting Nadam optimizer improvesthe training performance while changing the loss weights does not have a positive effect on the  accuracy. <br />

This deep network is better than other alternative deep learning networks in applications such as computer vision and Natural Language Processing (NLP) asit can mitigate the error rate significantly and hence improvenetwork performances. YOLO is refreshingly simple (see Figure 1), a single convolutional network simultaneously predictsmultiple bounding  boxes and class probabilities for thoseboxes. YOLO trains on full images and directly optimizesdetection performance. This unified model has several benefitsover traditional methods of object detection. <br />


<p align="center">
<img width="650" height="200" alt="d1" src="https://user-images.githubusercontent.com/71558720/99200627-676fc580-2774-11eb-9e89-5092efc722a8.png">
</p>
<p align="center">
<em>Fig.1: The YOLO Detection System. Processing images with YOLO is simple and straightforward. The system (1) resizes the input image to 448×448, (2) runs a single convolutional network on the image, and (3) thresholds the resulting detections by the model’sconfidence</em>
</p> <br /> 

#### Problem formulation:  
The main scientific questions of thispaper are as follow: <br /> 

+ To explore how the loss value of the YOLO v2 changes byintroducing a new anchor box set in the design method.
+ Changing the training optimizer factor to improve the network performance.
+ To test the effect of different components of the loss function (by changing their weights) on the overall accuracyof the network. <br /> 




































