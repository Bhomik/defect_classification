# defect_classification
This code is for classifying healthy vs defective material.


Solution -

1.) Divide the dataset into training and validation sets. I put 25 defective images and 35 healthy images in validation dataset. In total, there are 60 validation images and 190 training images to check the accuracy of algorithm on validation dataset. However, the validation set is small enough and may not be representing the test set in this case.

2.) Applied augmentation to training images, so that we can have different variations while training as our dataset is quite small.

3.) Made a small CNN so that it can be deployed on a constrained hardware.

4.) Applied sigmoid layer at the end as its a binary classification problem.

5.) Tuned the hyperparameters so that good accuracy can be achieved. Input images are resized to 300x300 and other hyperparameters are tuned as per the target.

6.) Saved the best model, which gives the highest accuracy on validation set defined and plot the accuracy and loss graphs for 50 epochs.
