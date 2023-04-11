#Hannah Lapp

#Make sure you have activated your deeplabcut conda environment
#conda activate DEEPLABCUT

#navigate to the AMBER-pipeline folder before runing this script


print('Importing packages')
import os
import sys
import deeplabcut
import PhenoPickleRaw as PhenoPickleRaw
import join_dam_pup


def main(argv):
    #set path to dlc projects and video directories
    dam_config = './dam-single-animal-Hannah-2022-05-26/config.yaml'

    pup_config = './pup-nine-pt-Hannah-2022-06-05/config.yaml'

    print('heres argv', argv)

    video_directory = argv[1]

    print()
    print()
    print('Running AMBER dam pose estimation')
    #Analyze videos for dam pose estiamtion
    deeplabcut.analyze_videos(dam_config, [video_directory], save_as_csv = True)

    if 'skip_create_videos' not in argv:
        #Create videos to check detections for dam tracking
        print()
        print('Creating dam tracking videos')
        deeplabcut.create_labeled_video(dam_config, [video_directory], save_frames = False)
    else:
        print()
        print(' Skipping create tracking videos')

    #Analyze videos for pup pose estimation
    print()
    print('Running AMBER pup pose estimation')
    deeplabcut.analyze_videos(pup_config, [video_directory], auto_track=False)

    if 'skip_create_videos' not in argv:
        #Create videos to check pup tracking
        print()
        print('Creating pup detections videos')
        deeplabcut.create_video_with_all_detections(pup_config, [video_directory])
    else:
        print()
        print(' Skipping create tracking videos')

    #Unpickle the raw pup dections files to convert to csv
    #this uses the PhenoPickle.Raw.py script

    pickle_dir = video_directory.replace("'", "")
    print()
    print('Unpickling pup pose estimation files')
    #os.system("python PhenoPickleRaw.py -input_directory:pickle_dir")
    PhenoPickleArgs = ['x', '-input_directory:' + pickle_dir]
    PhenoPickleRaw.main(PhenoPickleArgs)

    print()
    print('Joining dam and pup pose estimation files')
    join_dam_pup.main(argv)

def run_amber_pose_estimation():
    sys.exit(main(sys.argv))


if __name__ == '__main__':
    run_amber_pose_estimation()


