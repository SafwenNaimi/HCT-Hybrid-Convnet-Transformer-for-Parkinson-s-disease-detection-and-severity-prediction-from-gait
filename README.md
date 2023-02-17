# HCT-Hybrid-Convnet-Transformer-architecture-for-PD-detection-and-severity-prediction

## Prerequisites
Python 3.8 or later with all requirements.txt dependencies installed. To install run:

`python -m pip install -U pip

pip install -r requirements.txt`


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


