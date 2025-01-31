# ASL Alphabet Recognition using Computer Vision

### Overall Dataflow

#### 01 Preprocess ASL Alphabet

First we download the data from 'grassknoted/asl-alphabet' from kaggle.
The data is organized into train and test folders. The train folder contains
subfolder for each letter of the alphabet and some special characters. Next, we
move the data to the current working directory for preprocessing. Using opencv,
we can read and display the data. We use the mediapipe hands model to extract the
'landmarks' from the data. Each 'landmark' is represented by (x, y, z) coordinates.
We can build a dataset using these points and save it using pickle. Before we do
this we can define a mapping between the characters and numerical labels. The
numerical labels are arbitrary, but are just easier to work with. For each data point,
we append the x and y coordinates and the label mapped to its respective numerical represenation.
Lastly, we ensure that the data is correct by comparing the shape of X (the data),
and y (the labels).

#### 02 Training ASL Alphabet Classifier on Preprocessed Data

The next step is the train the model. First, we load the data
that we saved as a pickle file and split the data into train and test sets. Since the
test set of the original data only contained 1 example for each class, we will split
the dataset that we generated into an 80/20 split. It is important to specify shuffle
to True and to stratify on the labels to reduce any potential bias. After testing using
a RandomForestClassifier, XGBoost Classifer and a MLP, the RandomForestClassifier
yielded the best perfomance of 99.01% accuracy.

#### 03 Find Camera Index

This file just prints out the first avaliable index of the camera avaliable running on the
current device. The index can be passed into opencv VideoCapture object like this, cv2.VideoCapture(index).

#### 04 Testing ASL Alphabet Translator

To visualize this data use opencv to capture video from the user webcam. We extract 'landmarks'
from the camera in the same way we did when creating our dataset and pass it to our trained model. The model outputs the prediction
which we can display on the webcam feed along with a bounding box defined by the min
and max of the all the landmarks.

### Python Version

```
python=3.11.11
```

### Install Dependencies

```
pip install -r requirements.txt
```

Also download PyTorch from [here](https://pytorch.org/get-started/locally/).

### Challenges

1. When getting data from the mediapipe model, something some of the landmarks were not detected, resulting in nonhomogeneous data.
   - solved this by validating that each data point resulted in 42 values (21 x values and 21 y values) and discarding data that doesn't meet this criteria
2. The camera cannot be detected by opencv when running the program in WSL.
   - cloned project to windows file system

### Further Considerations

1. The dataset contains images of "nothing" (images with no hand in it), which would result in no landmarks detected, but it it important to detect "nothing" or not?
2. The best model so far is RandomForestClassifier with a performance of 99.01% but doesn't seem to perform well on data from the camera. Possibly alternaive neural network architectures, optimzers, or loss functions could yield better performace or maybe finetuning RandomForestClassifier or XGBoost.
