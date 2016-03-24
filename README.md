#### Deep Poisson Factor Modeling

The Matlab Code for the NIPS 2015 paper "Scalable Deep Poisson Factor Modeling"

Ricardo Henao, Zhe Gan, James Lu and Lawrence Carin

Department of Electrical and Computer Engineering, Duke University

#### Intro

These scripts implement Deep Poisson Factor Modeling (DPFM), from the 2015 NIPS paper with the same title. The source code is made publicly available for reproducibility purposes, it is not optimized for speed, minimally documented but fully functional.

Parts of the code are implemented in C via mex interface, these scripts need to be compiled and are platform dependent.

Change data and results paths to fit your needs.

#### Disclaimer

The code comes with no guarantee, it is scarcely documented, does not have parameter checks, have been tested only in Matlab R2015. If you find any bug or just have a suggestion please do not hesitate to contact the first author. Permission is granted to use and modify the code at your own risk.

#### Citing DPFM

Please cite the associated NIPS paper in your publications if this code helps with your research:

    @inproceedings{DPFA_ICML2015,
      Author = {R. Henao, Z. Gan, J. Lu, and L. Carin},
      Title = {Deep Poisson Factor Modeling},
      booktitle={NIPS},
      Year  = {2015}
    }

#### Data
	
The 20news and RCV2 dataset we used can be downloaded from:

https://drive.google.com/drive/u/0/folders/0B1HR6m3IZSO_WkdFWTdVU1JjMWM

#### Contact

r.henao >>at<< duke.edu