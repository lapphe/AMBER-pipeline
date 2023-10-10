# AMBER tutorial

## Before getting started
You can use the provided example videos or your own videos to complete this tutorial. The example video files can be downloaded from our OSF repository (https://osf.io/e3dyc/) from the "example_videos" folder. <br>
![OSF example videos](https://github.com/lapphe/AMBER-pipeline/assets/53009913/f2d20ad6-fbf3-4b65-bd59-a5a92354af95)
<br> 
<br>
If you are using your own videos, make sure all the videos you want to analyze at once are in their own  directory on your computer. The pose estimation script will run pose estimation steps on all the video files in the folder you give it. Move any videos you do not want run to another location. <br>
<br>If you are wondering about the best way record videos to be compatible with AMBER, check out the [Video Recording information page](https://github.com/lapphe/AMBER-pipeline/wiki/Video-Recording). <br>
<br>
Make sure you have followed the instructions for installing and setting up: <br>
--1 DeepLabCut <br>
--2 SimBA <br>
--3 AMBER files from this github repository<br>
--4 Behavior classifiers from the OSF repository <br>
<br>
Detailed installation and set up instructions can be found [here.](https://github.com/lapphe/AMBER-pipeline/wiki/Installations-and-set-up)

# Pose estimation
The AMBER_pose_estimation.py script will run your videos through all pose estimation and post-pose estimation steps required for all videos in the video folder. It will then prepare files for use in SimBA. <br>
<br>The script will automatically run the following steps:<br>
--1 Pose estimation for dams for all videos using DeepLabCut and the AMBER dam pose estimation model<br>
--2 Create videos to check dam tracking<br>
--3 Pose estimation for pups for all videos using DeepLabCut and the AMBER pup multi-animal pose estimation model<br>
--4 Create videos to check pup detections<br>
--5 "Unpickle" pup detection files to convert to csv <br>
--6 Join and reformat pup and dam pose estimation output so it is ready to use with SimBA 
<br>

## Run pose estimation steps  
<br> 
**-1 Open the windows command prompt with administrator privileges**
<br>
<br>
**2 Activate your deeplabcut conda environment** <br> 
``conda activate DEEPLABCUT``
<br>
<br>
**3 Move to the AMBER-pipeline directory** <br>
Change your current directory so you are in the AMBER-pipeline directory containing all the files downloaded when you cloned the AMBER repository using `cd /d path/to/directory` on windows
<br> e.g. if the AMBER-pipeline folder is located on the desktop: `cd /d C:\Desktop\AMBER-pipeline`
<br>
<br> 
**4) Make sure all the videos you want to run are located in a single folder** <br>
They can be anywhere on your computer -they do not need to be in the AMBER-pipeline folder. Copy the address of the folder containing the videos to run. <br>
   _Note: in windows, you can copy the directory path by right-clicking on the folder name in the file explorer and selection “Copy address as text”. You can then paste it in the the command window_
<br>
<br>
**5) Run pose estimation steps** <br>
To run pose estimation, you will enter “python”, the script name,  followed by the path to the directory where your videos are location.
<br> 
e.g. `python AMBER_pose_estimation.py C:\Desktop\example_videos`
<br> Press enter to execute the command.
<br>
<br>
The deeplabcut files will appear in the same directory as your videos. There will also be **two new folders created**: <br>
<br>
The first folder, _pose_estimation_videos_, contains the video with the labeled dam and pup points to check model performance. These videos have been move to a separate directory to make importing videos into SimBA easier later on. <br>
<br>
The second folder, _AMBER_joined_pose_estimation_, contains the reformatted and combined pose estimation files with dam and pup tracking that are used during behavior classification in SimBA. 
<br> 
<br>
6) Check your pose estimation videos 
Check the labeled videos to ensure that the tracking looks good before proceeding to behavior classification. If the pose estimation models are not performing well, you may need to label additional frames from your videos and retrain the dam or pup models. Note that pose estimation does not need to be perfect to get accurate behavior classification, but major tracking mistakes will reduce the performance of the classifiers. 
<br>
<br>
If you feel confident in the pose estimation model performance on your videos and want to skip the “create tracking videos” step, you can add the “skip_create_videos” argument when you run the script.  <br>
e.g. `Python AMBER_pose_estimation.py C:\Desktop\hannah_test_short skip_create_videos`
<br>
<br>
**7) Exit your deeplabcut conda environment** <br>
`conda deactivate`

# Behavior classification <br>
Behavior classification is performed in SimBA using the preconfigure AMBER_SimBA_project. <br>

**1) Start the SimBA conda environment and open simba**
<br>
`conda activate simbaenv` <br>
`simba` <br>
<br>
<br>
**2) Load the SimBA_AMBER_project using the SimBA GUI.** <br>
The project config file is found in _AMBER-pipeline/SimBA_AMBER_project/project_folder/project_config.ini_. From here, you can follow the SimBA user [guide for analyzing new videos](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md). Below is an overview of the steps. <br>
<br>
<br>
**3) Import videos** <br> 
These should be the same videos you performd pose estimation on. You can select the "Import SYMLINK" box to use symbolic links to the videos instead of copying the full videos into the SimBA project.
<br>
<br>
**4) Import tracking data** <br>
These csv files can be found in the “AMBER_joined_pose_estimation” folder created during pose estimation.  <br>
<br>
<br>
**5) Set video parameters** <br>
As described in the [Video recording section](https://github.com/lapphe/AMBER-pipeline/wiki/Video-Recording "Video recording for AMBER"), we used the know distance of the food hopper, which is positioned about halfway of the depth of the cage (see images on under Video Recording). As a side-view recording, distances calculated will not be completely accurate since the actual distance varies depending on the location of the animal in the cage. However, setting these known distances helps account for variation in recording resolution and the distance from the cage to the front of the cage. Ifyour cage set up is different, you can select a different know distance visible in your videos, although we suggst selecting something that is about at the mid point of the cage depth. <br>
<br>
<br>
**6) Skip outlier correction** <br>
We recommend skipping outlier correction because this step relies on body-length distance across all frames to perform these calculations, which is influenced by the dramatic differences in body length when the dam is near the front versus back of the cage.<br>
(Don’t forget to actually tell SimBA to skip this step on the Outlier Correction tab! This will ensure the csv files are copied over to the new location and can be used ifor feature extraction in the next step.) <br>
<br>
<br>
**6) Extract features** <br>
Select "Apply user-defined feature extraction script" and use the customized AMBER feature extraction script. This script is located in AMBER-pipline/SimBA_AMBER_project/AMBER_feature_extraction/amber_feature_extraction.py<br>
![extract features](https://user-images.githubusercontent.com/53009913/232091989-cd38972c-6d97-4248-b5c8-2384bc7938e5.png)
<br>
_Note: This step can take a long time for long videos. The convex hull and back circle fitting calculations take a lot of computational time, but are among the most important features for several behavioral classifiers. For an hour long video recorded at 30fps, this step takes about 25 minutes per video, however, run time will vary depending on your computer specs._ <br.
<br>
<br>
**(Skip the "Label behavior" and "Train machine models" steps. Those steps are used for creating new behavior classifier models. We will use previously created models)** <br>
<br>
**7) Run the machine models** <br>
It’s a good idea to [validate the provided models on your videos](https://github.com/sgoldenlab/simba/blob/master/docs/validation_tutorial.md) on your videos* and determine a good discrimination threshold for each classifier before running the models on all of your videos. Below are discrimination thresholds that work well for the example videos, but you should confirm performance with your own videos. <br>
    Nest attndance: 0.5<br>
    Active nursing: 0.4 <br>
    Passive nursing: .2 <br>
    Self-directed grooming: 0.25 <br>
    Eating: 0.28 <br>
    Drinking: 0.22 <br>
<br>
<br>
**8) Analyze all of your videos** <br>
Find the models (they were moved to _AMBER-pipline/SimBA_AMBER_project/models_ during set up) and then enter the discrimination threshold and minimum bout length for analysis. Click “Run models”. <br>
<br>
### Congratulations, you now have maternal behavior annotations!

SimBA provides several tools for post-classification analysis and [visualizations](https://github.com/sgoldenlab/simba/blob/master/docs/visualizations_tutorial.md) that can be used with your data. Or, you can use the csv files found in _?AMBER_SimBA_project/project_folder/csv/machine_results_ that contain behavior lables for each frame along with the extracted features and pose estimation coordinates for all body parts. 
