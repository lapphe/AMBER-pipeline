# Automated Maternal Behavior during Early life in Rodents pipeline

Mother-infant interactions during the early postnatal period are critical for infant survival and the scaffolding of infant development. Rodent models are used extensively to understand how these early social experiences influence neurobiology across the lifespan. However, methods for measuring postnatal dam-pup interactions typically involve time-consuming manual scoring, vary widely between research groups, and produce low density data that limits downstream analytical applications. To address these methodological issues, we developed the Automated Maternal Behavior during Early life in Rodents (AMBER) pipeline for quantifying home-cage maternal and mother-pup interactions using open-source machine learning tools from side-view recordings. We hope this pipeline will allow non-programmers access to advanced machine learning tools to improve standarization of maternal behavior measures across studies. <br>
**Check out our [bioxriv preprint](https://www.biorxiv.org/content/10.1101/2023.09.15.557946v1)** <br>
_AMBER pose estimation models are currently optimized for Long-Evans rats, but we are currently expanding this to several other mouse and rat strains to ensure AMBER can be used with all rodent animal models.<br> Please get in touch if you have videos or frames you would like to contribute to this effort._ <br>

# How does the pipeline work?
The AMBER pipeline uses side-view video recordings of dam and pups in the home cage. The first steps of the pipeline involve extracting keypoints (labeled body points) from dams and pups using provided DeepLabCut models. Next, keypoint coordinate information produced by the deeplabcut models from dam and pups is joined together and formatted. This data is then imported into SimBA and 218 features are calculated (e.g. the distance between the centroid point of the dam and the centroid point of the pups). The behavior predictions are generated using behavior classifiers trained to recognize seven different maternal behaviors from the features. The result is behavior annotations for every frame of the video. <br>
![Figure 1](https://github.com/lapphe/AMBER-pipeline/assets/53009913/0b442b48-a238-4fd2-a065-322480bc6060)
<br>
## Behavior definitions
See our [annotations wiki page](https://github.com/lapphe/AMBER-pipeline/wiki/Behavior-annotation-information) for an overview of how behaivors are operationalized. Our full [AMBER ethogram guide](https://docs.google.com/document/d/1YB2kZJxlYC2BvaRYZfWjrgPuMP7aJ4sWBeQbfwD3xLA/edit?usp=sharing) provides detailed information on how behavior annotations were made to train each of the behavior classifiers in SimBA. <br>
AMBER behavior classifiers for SimBA can be downloaded from OSF: https://osf.io/e3dyc/ <br>

# Example useage
Check out the [tutorial](https://github.com/lapphe/AMBER-pipeline/blob/main/tutorial.md) for step by step instructions for using the AMBER pipline on example videos or your own videos.

# Installation and set up
See the [installation wiki page](https://github.com/lapphe/AMBER-pipeline/wiki/Installations-and-set-up) for detailed information. <br>
We recommend installing deeplabcut and SimBA in different anaconda environments as these software programs have different dependency requirements. 
This workflow has been tested on Windows 11. <br>
## Deeplabcut
Our models were generated with DeepLabCut version 2.3.0, but should be compatible with newer versions. 
Please follow the instructions for installing DeepLabCut provided on their [installation documentation page](https://deeplabcut.github.io/DeepLabCut/docs/installation.html). 
## SimBA
Behavioral classifiers were generated using SimBA version 1.65.5. Please follow the [SimBA installation guide.](https://github.com/sgoldenlab/simba/blob/master/docs/installation.md)
 <br> 

### Install the circle-fit package
The AMBER custom feature extraction script uses functions from the circle-fit package. This package is not automatically added during SimBA installation, so it needs to be installed separately in the SimBA environment. 

## Download AMBER files
1. The pose estimation models and scripts required for the AMBER pipeline can be found in this github repository. You should download all scripts as a zip file or by cloning the repository. <br>
`git clone https://github.com/lapphe/AMBER-pipeline.git` <br>
![git download small](https://user-images.githubusercontent.com/53009913/232549974-0763b7c2-0af1-4b00-8d5c-d7372131be60.png)

2. You will need to update the AMBER SimBA project config file to reflect the specific location (path) of the project files on your computer. To do this, find the project_config.ini file located in AMBER-pipeline/SimBA_AMBER_project/project_folder/.

3. Update the **project_path** (line 2) and **model_dir** (line 8) int he config file to reflect the current location. 
4. The behavioral classifiers are large files and cannot be stored on github. Please download them from our [OSF repository](https://osf.io/e3dyc/). Move the behavior classification models to the “models” folder in AMBER-pipeline/SimBA_AMBER_project/models.<br>
![OSF download small](https://user-images.githubusercontent.com/53009913/232550321-32c23eca-334e-4c9e-a762-39e07590a962.png)

# References and resources
[DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) <br>
[SimBA](https://github.com/sgoldenlab/simba/tree/master)<br>
[AMBER bioxriv preprint](https://www.biorxiv.org/content/10.1101/2023.09.15.557946v1)

