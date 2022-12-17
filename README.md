# Custom-Dataset-Training-with--DE⫶TR
Custom Dataset Training pipeline using Pytorch and Meta's object detection model [DE⫶TR](https://github.com/facebookresearch/detr). 

# DE⫶TR Detection Transformer

**DETR** (**DE**tection **TR**ansformer) is object detection model developed by [Meta AI](https://ai.facebook.com/).

[Paper](https://ai.facebook.com/research/publications/end-to-end-object-detection-with-transformers) and [repository](https://github.com/facebookresearch/detr) here.

# Dataset

[VisDrone2019](http://aiskyeye.com/home/) Dataset is an object detection and tracking dataset that consist of around 10k images. [To read more...](http://aiskyeye.com/home/)

Images from dataset
| 1 | 2 |
|------|------|
|<img src="images/2.jpg">|<img src="images/3.jpg">|

# Project Structure

```
 Custom Dataset Training with DETR
 ├───detr
 │   ├───datasets
 │   ├───models 
 │   └───util     
 ├───VisDrone2019
 │   ├───Train
 │   │   ├───annotations
 │   │   └───images
 │   ├───Validation
 │   │   ├───annotations
 │   │   └───images
 │   └───ValidationOld
 │       ├───annotations
 │       └───images
 └───weights
```

# Training Pipeline

- First you need an custom dataset class or you just can use my VisDroneDatasetClass. 
- Secondly just adjust the transformation functions on utils.py
- Finally add those changes you made in other files onto run.py file and run. 