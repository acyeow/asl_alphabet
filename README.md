# ASL Alphabet Recognition using Computer Vision

### Python Version

```
python=3.11.11
```

### Install Dependencies

```
pip install -r requirements.txt
```

### Challenges

1. When getting data from the mediapipe model, something some of the landmarks were not detected, resulting in nonhomogeneous data.
   - solved this by validating that each data point resulted in 42 values (21 x values and 21 y values) and discarding data that doesn't meet this criteria
2. The camera cannot be detected by opencv when running the program in WSL.
   - cloned project to windows file system

### Further Considerations

1. The dataset contains images of "nothing" (images with no hand in it), which would result in no landmarks detected, but it it important to detect "nothing" or not?
2. The best model so far is RandomForestClassifier with a performance of 99.01% but doesn't seem to perform well on data from the camera. Possibly neural networks could yield better performace or maybe finetuning RandomForestClassifier or XGBoost.
