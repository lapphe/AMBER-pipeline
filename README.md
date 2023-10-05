# Automated Maternal Behavior during Early life in Rodents pipeline
Mother-infant interactions during the early postnatal period are critical for infant survival and the scaffolding of infant development. Rodent models are used extensively to understand how these early social experiences influence neurobiology across the lifespan. However, methods for measuring postnatal dam-pup interactions typically involve time-consuming manual scoring, vary widely between research groups, and produce low density data that limits downstream analytical applications. To address these methodological issues, we developed the Automated Maternal Behavior during Early life in Rodents (AMBER) pipeline for quantifying home-cage maternal and mother-pup interactions using open-source machine learning tools from side-view recordings. We hope this pipeline will allow non-programmers access to advanced machine learning tools to improve standarization of maternal behavior measures across studies. 
**AMBER provided pose estimation models are currently optimized for Long-Evans rats, but we are currently expanding this to several other mouse and rat strains to ensure AMBER can be used with all rodent animal models. Please get in touch if you have videos or frames you would like to contribute to this effort.**

## How does the pipeline work?
The AMBER pipeline uses side-view video recordings of dam and pups in the home cage. The first steps of the pipeline involve extracting keypoints (labeled body points) from dams and pups using provided DeepLabCut models. Next, keypoint coordinate information produced by the deeplabcut models from dam and pups is joined together and formatted. This data is then imported into SimBA and 218 features are calculated (e.g. the distance between the centroid point of the dam and the centroid point of the pups). The behavior predictions are generated using behavior classifiers trained to recognize seven different maternal behaviors from the features. The result is behavior annotations for every frame of the video. 
[insert pipeline overview]

##Behavior definitions
Below is a summary of how each behavior was operationalized. Our full [AMBER ethogram guide](https://docs.google.com/document/d/1YB2kZJxlYC2BvaRYZfWjrgPuMP7aJ4sWBeQbfwD3xLA/edit?usp=sharing) provides detailed information on how behavior annotations were made to train each of the behavior classifiers in SimBA. <br>
AMBER behavior classifiers for SimBA can be downloaded from OSF: https://osf.io/e3dyc/ <br>

#Installation and set up
We recommend installing deeplabcut and SimBA in different anaconda environments as these software programs have different dependency requirements. 
This workflow has been tested on Windows 11. 
# Deeplabcut
Our models were generated with deeplabcut version 2.3.0, but should be compatible with newer versions. 
Please follow the instructions for installing deeplabcut provided on their [installation documentation page](https://deeplabcut.github.io/DeepLabCut/docs/installation.html). 
We recommend testing new installations using the deeplabcut [test script.](https://www.youtube.com/watch?v=IOWtKn3l33s&themeRefresh=1)
***
# SimBA
Behavioral classifiers were generated using SimBA version 1.31.2, have been successfully tested in SimBA 1.55.6, and should work with newer versions as well. Please follow the [SimBA installation guide.](https://github.com/sgoldenlab/simba/blob/master/docs/installation.md)
 <br> 
      _Note: with SimBA version 1.55.6, please also run_
       `pip install numba==0.52.0`
       _after SimBA installation_
<br>
### Install the lsq-ellipse package
The AMBER custom feature extraction script uses functions from the lsq-ellipse package. This package is not automatically added during SimBA installation, so it needs to be installed separately in the SimBA environment. 
<br> To do this, activate the SimBA anaconda environment <br>
    e.g. `conda activate simbaenv`
   <br>
    Then install the lsq-ellipse package: 
   <br> `pip install lsq-ellipse`

***

# Download AMBER scripts
1.The pose estimation models and scripts required for the AMBER pipeline can be found in this github repository and on our [OSF repository](https://osf.io/e3dyc/). You should download all scripts as a zip file or by cloning the repository. <br>
`git clone https://github.com/lapphe/AMBER-pipeline.git`
<br>
![git download small](https://user-images.githubusercontent.com/53009913/232549974-0763b7c2-0af1-4b00-8d5c-d7372131be60.png)
<br>
<br>
2. You will need to update the AMBER SimBA project config file to reflect the specific location of the project files on your computer. To do this, find the project_config.ini file located in AMBER-pipeline/SimBA_AMBER_project/project_folder/.
<br>
<br>
3.Double click to open the file in a text editor. The **project_path **(line 2) and **model_dir** (line 8) need to be updated to reflect the current location. 
<br>
For example, if the AMBER-pipeline folder is located on your Desktop, changing the path might look similar to the image below:
<br>

![simba config small](https://user-images.githubusercontent.com/53009913/232550858-fe426eba-d5cf-428c-bc09-418431817cfe.png)
<br>
<br>
4.The behavioral classifiers are large files and cannot be stored on github. Please download them from our [OSF repository](https://osf.io/e3dyc/). 
<br> 
![OSF download small](https://user-images.githubusercontent.com/53009913/232550321-32c23eca-334e-4c9e-a762-39e07590a962.png)
<br>
<br>
5. Unzip/extract the behavior model files (e.g. using 7 zip)
<br>
<br>
6. Move the behavior classification models to the “models” folder in AMBER-pipeline/SimBA_AMBER_project/models. This is inside the directory cloned from the AMBER github repository

#Example usage
Example video files are provided in the folder......

The AMBER_pose_estimation.py script will run your videos through all pose estimation and post-pose estimation steps require to prepare files for use in SimBA. 


## To run the script: 
<br> 1) Open the windows command prompt with administrator privileges
<br>
<br> 2) Activate your deeplabcut conda environment: <br> 
``conda activate DEEPLABCUT``
<br>
<br> 3) Change your directory so you are in the AMBER-pipeline directory containing all the files downloaded when you cloned the AMBER repository using `cd /d path/to/directory` on windows
<br> e.g. if the AMBER-pipeline folder is located on the desktop: `cd /d C:\Desktop\AMBER-pipeline`
<br>
<br> 4) Make sure all the videos you want to run are located in a single folder (anywhere on your computer). This script will run pose estimation steps on all the video files in the folder you give it. Make sure any videos you do not want run are moved to another location. Copy the address of the folder containing the videos to run. <br>
_Note: in windows, you can copy the directory path by right-clicking on the folder name in the file explorer and selection “Copy address as text”. You can then paste it in the the command window_
<br>
<br>
<br>
<br> 5) To run pose estimation, you will enter “python”, the script name,  followed by the path to the directory where your videos are location.
<br> 
e.g. `python AMBER_pose_estimation.py C:\Desktop\hannah_test_short`
<br> Press enter to execute the command.
<br>
<br>  The script will automatically run the following steps:<br>
----1 Pose estimation for dams for all videos<br>
----2 Create videos to check dam tracking<br>
----3 Pose estimation for pups for all videos<br>
----4 Create videos to check pup detections<br>
----5 "Unpickle" pup detection files to convert to csv <br>
----6 Join and reformat pup and dam pose estimation output so it is ready to use with SimBA 
<br>
_Note: The above steps can also be completed separately using the individual files supplied with AMBER_
<br>
<br>
The deeplabcut files will appear in the same directory as your videos. There will also be **two new folders created**: <br>
<br>
First, the _pose_estimation_videos_ folder contains the video with the dam and pup track points to check model performance. These videos have been move to a separate directory to make importing videos into SimBA easier later on. <br>
<br>
Second, the _AMBER_joined_pose_estimation_ folder contains the reformatted pose estimation files with dam and pup tracking that should be used during behavior classification in SimBA. 
<br> 
<br>
6) Check your pose estimation videos to ensure that the tracking looks good before proceeding to behavior classification. If the pose estimation models are not performing well, you may need to label additional frames from your videos and retrain the dam or pup models. 
<br>
<br>
If you feel confident in the pose estimation model performances and want to skip the “create tracking videos” step, you can pass “skip_create_videos” when you run the script. <br>
e.g. `Python AMBER_pose_estimation.py C:\Desktop\hannah_test_short skip_create_videos`
<br>
<br>
7) Exit your deeplabcut conda environment <br>
`conda deactivate`





#References and resources
The first part of the pipeline uses single animal and multi-animal DeepLabCut. DeepLabCut provides extensive resources for installation, runing, and managing DeepLabCut projects. 
SimBa
