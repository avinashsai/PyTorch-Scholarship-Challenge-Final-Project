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

!wget -O flower_data_orginal_test.zip \"https://www.dropbox.com/s/da6ye9genbsdzbq/flower_data_original_test.zip?dl=1
!unzip flower_data_orginal_test.zip  -d /testoriginal
```

# Results:

Accuracy of different models is reported on both the datasets:

 Model | Google Dataset | Original Dataset |
 
 Densenet161 Model-1 | 62.16216216216216 | 90.96459096459097|
 
 Densenet201 Model-1 | 60.36036036036037 | 90.84249084249085 |
 
