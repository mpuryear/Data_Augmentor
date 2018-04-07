# Data_Augmentor

Data augmentor is a simple way to use the Keras built in ImageDataGenerator class

### Prerequisites

What things you need to install the software and how to install them

```
Keras and numpy are the only required packages
```

### Installing



Keras installation process

```
https://github.com/ignaciorlando/skinner/wiki/Keras-and-TensorFlow-installation
```


Numpy installation process 

```
https://scipy.org/install.html
```

## Running the test

To run the only test provided, uncomment the final line in augment_data.py

```
python augment_data.py
```

### Test output
When running with the test function uncommented. 20 augmented images will be created and placed into the already created folders. 

10 images will be placed into the ./data_aug/with/  folder
10 images will be placed into the ./data_aug/without/  folder

### Pretest file structure /  Post test file structure
![Pre/Post file structures](https://i.imgur.com/GCKSkl8.png)



## Deployment

To use this in a live system.

Python:
```
from augment_data import augment_data

x_aug, y_aug = augment_data(...)
```

## Notable things also mentioned inside the augment_data.py

The generated images are given a number that references the index number in the provided array.

There are additional parameters that a user can fine tune inside the augment_data() function where a variable datagen is created. In a comment just above this function call there is a url provided to the Keras function and its adjustable parameters. 

This function is not expected to handle any of the folder creation, or image preprocessing. That should be done externally.
