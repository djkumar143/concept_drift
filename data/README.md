# Datasets used

## elec2
The original dataset used in this project.
<br>

## elec2_past_data
Dataset formed from elec2 with first 35000 records for training the base model

## elec2_real_time_data
Dataset with remaining rows after first 35000 records. Will be used for real time streaming

## elec2_real_time_features
Datasets consists of only features and no class, each row associated with an event.
<br>
Will be used for real time prediction of labels.

## elec2_true_labels
Dataset consisting of only true labels associated with same events.