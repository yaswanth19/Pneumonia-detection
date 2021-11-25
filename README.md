# Pneumonia
Pneumonia is the leading cause of death among young children and one of the top mortality causes worldwide. The pneumonia detection is usually performed through examine of chest X-Ray radiograph by highly trained specialists. This process is tedious and often leads to a disagreement between radiologists. Computer-aided diagnosis systems showed potential for improving the diagnostic accuracy. In this work, we develop the computational approach for pneumonia regions detection based on single-shot detectors, squeeze-and-extinction deep convolution neural networks, augmentations and multi-task learning. The proposed approach was evaluated in the context of the Radiological Society of North America Pneumonia Detection Challenge, achieving one of the best results in the challenge. Our source code is freely available here.

### Dataset
The labelled dataset of the chest X-Ray (CXR) images and patients meta data was publicly provided for the challenge by the US National Institutes of Health Clinical Center. The dataset is available on kaggle platform.The following figure shows the sample images from the data set for both the classes NORMAL and PNEUMONIA.
![2021-11-24 (3)](https://user-images.githubusercontent.com/82788246/143465784-5235d217-868d-4fc9-b2c1-1f7146734d0e.png)

>### Metrics
>Since this was a classification challenge the metrics used was accuracy

### Data Augmentations
Since the data size was small the following data augmentations were peformed
1.)Horizontal Flip
2.)Zoom Range
3.)Shear Range

## Model
I have built a custom CNN model eventhough it was a shallow architecture model with two layers it gave pretty good results with hyperparameter tuning.The code for the model is :- 
```python
model=keras.Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)))
model.add(MaxPool2D(2,2))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(activation = 'relu', units = 128))
model.add(Dense(activation = 'sigmoid', units = 1))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
```
## Hyperparameter Tuning
```python
cnn_model = model.fit_generator(training_set,
                         steps_per_epoch = 163,
                         epochs = 1,
                         validation_data = validation_generator,
                         validation_steps = 624)
```                         
                         
## Results
```bash
loss: 0.3932
accuracy: 0.8286 
val_loss: 0.3794 
val_accuracy: 0.9375
```
>## On Test Data
```bash
loss: 0.31164586544036865 
accuracy: 0.8573718070983887
```
The results are overwhelmings considering the fact that the original dataset was small and we have not used Transfer Learning with pre-trained models.
