# HCT-Hybrid-Convnet-Transformer-architecture-for-PD-detection-and-severity-prediction

## Prerequisites
Python 3.8 or later with all requirements.txt dependencies installed. To install run:


`python -m pip install -U pip`

`pip install -r requirements.txt`


## Dataset
The dataset used in the paper is from Physionet. It can be downloaded from (1). A sample from each group of researchers contributed in the dataset is available in the subfolder data. 
Ga, Ju or Si – indicate the study from which the data originated:
* Ga - Galit Yogev et al (2) (dual tasking in PD; Eur J Neuro, 2005)
* Ju – Hausdorff et al (3) (RAS in PD; Eur J Neuro, 2007)
* Si - Silvi Frenkel-Toledo et al (4) (Treadmill walking in PD; Mov Disorders, 2005)
* Co or Pt: Control subject or a PD Patient

We also included demographics.xls that contains all the information about each example in the data.
Each line in demographics.xls contains 19 columns:

* Column      1:   Time (in seconds)
* Columns   2-9:   Vertical ground reaction force (VGRF, in Newton) on each of 8
	  	  sensors located under the left foot
* Columns 10-17:   VGRF on each of the 8 sensors located under the right foot
* Column     18:   Total force under the left foot
* Column     19:   Total force under the right foot.

## Getting Started
To run experiments for our Two-class ConvNet-Transformer model, the entry point is `Two-Class_model.py` file.
The algorithm will generate the following output files:
├── output (dir)

    ├── train_classifier_month_day   
        ├── hour_minutes
	    ├──  model.json : JSON file of the model.               
	    ├──  res_pat.csv: results of accuracy, sensitivity and specificity by patients.
            ├──  res_seg.csv: results of accuracy, sensitivity and specificity by segments.	                
            ├──  training_i.csv: training/validation loss and accuracy for the i_th folder (i = [1..10]).   
	    ├──  weights_i.hdf5 : weights of the model for the i_th folder (i = [1..10]).   


To run experiments for our Multi-class ConvNet-Transformer model, the entry point is `Multi-Class_model.py` file.
The algorithm will generate the following output files:
├── output (dir)

    ├── train_severity_month_day   
        ├── hour_minutes
	    ├──  confusion_matrix.csv: Confusion matrix for severity prediction.
	    ├──  gt.csv: ground truth level for each patient.
	    ├──  pred.csv: prediction level for each patient.
	    ├──  model.json : JSON file of the model.               
	    ├──  res_pat.csv: results of accuracy by patients.
            ├──  res_seg.csv: results of accuracy by segments.	                
            ├──  training_i.csv: training/validation loss and accuracy for the i_th folder (i = [1..10]).   
	    ├──  weights_i.hdf5 : weights of the model for the i_th folder (i = [1..10]).   
	    

