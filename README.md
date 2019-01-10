# PyTorch-Scholarship-Challenge-Final-Project

# Getting Started

This is the final project of PyTorch Udacity Scholarship Challenge. Challenge is to perform image classification on 102 flowers dataset.
I have used different pretrained models and finetuned them to get better performance.

# Datasets

Download the dataset used for Training and Validation here using 

```
!wget -c https://s3.amazonaws.com/content.udacity-data.com/courses/nd188/flower_data.zip
```

Used 2 datasets for Testing. Download them using following commands(Must use same directory names while unzipping):


```
Google Dataset:

!wget -O google_test_data.zip "https://www.dropbox.com/s/3zmf1kq58o909rq/google_test_data.zip?dl=1"
!unzip google_test_data.zip -d /test

Original Dataset:

!wget -O flower_data_orginal_test.zip "https://www.dropbox.com/s/da6ye9genbsdzbq/flower_data_original_test.zip?dl=1
!unzip flower_data_orginal_test.zip  -d /testoriginal
```

# Results:

Accuracy of different models is reported on both the datasets:

| Model           |  Google Dataset      | Original Dataset |
 --- | --- | ---
DenseNet161 Model 1 | 62.16216216216216 | 90.96459096459097 
DenseNet201 Model 1  | 60.36036036036037 | 90.84249084249085
DenseNet201 Model 2  | 56.95695695695696 | 85.83638583638583
DenseNet201 Model 3  | 63.16316316316316 | 92.18559218559218
DenseNet201 Model 4  | 63.06306306306306 | 91.33089133089133
DenseNet201 Model 5  | 62.36236236236237 | 91.57509157509158
DenseNet201 Model 6  | 62.76276276276276 | 92.18559218559218
ResNet152 Model 1     |60.06006006006006 | 88.52258852258852
ResNet152 Model 2     |56.25625625625625 | 87.78998778998779
ResNet152 Model 3     |57.55755755755756 | 86.56898656898657
 VGG19 Model 1       | 48.048048048048045| 78.99877899877   
 ResNet50 Model 1    | 46.94694694694695 |75.21367521367522 

## Requirements

```
pytorch==0.4.0
torchvision
```

## Special Mention
@GabrielePicco for providing the test data
