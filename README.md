# yolo_weighted

1. prepare cross validation split setup
2. train on each slice of training data and gather feedback on the validation set: 
 - preset a confusion matrix for the noisy label.
 - calculate the probablity based on the match between predictions and noisy labels.
 - update label positions.
 - preset a threshold below which the labels should be removed for subsequent training. 
3. gather all the feedbacked validation set as the new dataset for training/cross validation at the next step.
