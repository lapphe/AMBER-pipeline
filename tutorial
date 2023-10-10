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
e.g. `python AMBER_pose_estimation.py C:\Desktop\example_videos`
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
6) Check your pose estimation videos to ensure that the tracking looks good before proceeding to behavior classification. If the pose estimation models are not performing well, you may need to label additional frames from your videos and retrain the dam or pup models. Note that pose estimation does not need to be perfect to get accurate behavior classification, but major trackin gmistakes will reduce the perforamnce of the classifiers. 
<br>
<br>
If you feel confident in the pose estimation model performances and want to skip the “create tracking videos” step, you can pass the “skip_create_videos” argument when you run the script. <br>
e.g. `Python AMBER_pose_estimation.py C:\Desktop\hannah_test_short skip_create_videos`
<br>
<br>
7) Exit your deeplabcut conda environment <br>
`conda deactivate`

### Open the AMBER SimBA project folder
<br>

1)Start the SimBA conda environment and open simba <br>
`conda activate simbaenv` <br>
`simba` <br>
<br>
2) Load the SimBA_AMBER_project using the SimBA GUI. The project config file is found in _AMBER-pipeline/SimBA_AMBER_project/project_folder/project_config.ini_. From here, you can follow the SimBA user [guide for analyzing new videos](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md). Below is a quick outline of the steps. <br>
<br>
3) **Import your Videos** <br>
<br>
4) **Import your joined pose estimation csv files** (found in the “AMBER_joined_pose_estimation” folder). <br>
<br>
5) **Set video parameters:** As described in the Video recording section, we used the know distance of the food hopper, which is positioned about halfway of the depth of the cage (see images on under Video Recording). As a side-view recording, distances calculated will not be accurate since the actual distance varies depending on the location of the animal in the cage. <br>
<br>
6)**Outlier correction:** We recommend skipping outlier correction because this step relies on body-length distance across all frames to perform these calculations, which is influenced by the dramatic differences in body length when the dam is near the front versus back of the cage.
(don’t forget to actually tell SimBA to skip this!) <br>
<br>
7)**Extract features:** Use the customize AMBER feature extraction script for feature extraction. This script is located in AMBER-pipline/SimBA_AMBER_project/AMBER_feature_extraction/amber_feature_extraction.py<br>
![extract features](https://user-images.githubusercontent.com/53009913/232091989-cd38972c-6d97-4248-b5c8-2384bc7938e5.png)
<br>
_Note: This step can take a long time for long videos. The convex hull and back ellipse fitting calculations take a particularly long time, but are among the most important features for several behavioral classifiers.  We are working on updating the script to make the calculations faster._ <br.
<br>
<br>
8)**Run the machine models:**
It’s a good idea to [validate the provided models on your videos](https://github.com/sgoldenlab/simba/blob/master/docs/validation_tutorial.md) on your videos* and determine a good discrimination threshold for each classifier before running the models on all of your videos. <br>
<br>
**To analyze all of your videos:** Find the models (they were moved to _AMBER-pipline/SimBA_AMBER_project/models_ during set up) and then enter the discrimination threshold and minimum bout length for analysis. Click “Run models”. <br>
<br>
<br>
Congratulations, you now have maternal behavior annotations! 
<br>
SimBA provides several other tools for post-classification analysis and [visualizations](https://github.com/sgoldenlab/simba/blob/master/docs/visualizations_tutorial.md) that can be used with your data.
