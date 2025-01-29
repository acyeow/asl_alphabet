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
2. The camera cannot be detected by opencv when running the program in WSL.

### Further Considerations

1. The dataset contains images of "nothing" (images with no hand in it), which would result in no landmarks detected, but it it important to detect "nothing" or not.
2. The best model so far is XGBoost with a performance of 99% but doesn't seem to perform well on data from the camera.
