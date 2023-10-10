"""
This script joins the dam and pup pose estimation files for the AMBER pipeline. 
The data frame is formatted to match the format expected by SimBA (single animal project)
and required for feature extraction with the AMBER feature extraction script

Last updated: October 5, 2023
Author: Hannah Lapp
"""
import os
import sys
import pandas as pd

def main(argv):
    """
    Process and merge pose estimation data for dam and pup videos.

    Given a video directory containing pose estimation CSV files for dam and pup videos,
    this function identifies files with specific naming patterns and merges the data
    based on frame number.

    Args:
        argv (list): Command-line arguments. the first argument is the name of this script
                     The second argument (argv[1]) should be the path to the directory 
                     containing pose estimation CSV files.

    Returns:
        None

    This function performs the following steps:
    1. Identifies CSV files for dam and pup videos based on the original video file name.
    2. Loads and preprocesses the data from these files.
    3. Merges the dam and pup data based on frame timestamps.
    4. Saves the merged data to CSV files in a subdirectory named 'AMBER_joined_pose_estimation'.

    Example:
        To process pose estimation CSV files in the 'videos' directory, run the script
        with the following command:

        >>> python join_dam_pup.py path/videos

    Note:
        - The script assumes specific naming patterns for dam and pup video files that are generated from the provided 
        pose estimation models.
        - The merged data is saved in a subdirectory named 'AMBER_joined_pose_estimation'.
    """
    print("Starting file joining...")
    video_directory = argv[1]
    all_files = os.listdir(video_directory)

    #given the video directory, find all files containing "DLC_resnet50_dam-single-animalMay26shuffle" and ending in ".csv"
    dam_files = []
    dam_keys = []
    for file in all_files:
        if 'DLC_resnet50_dam-single-animalMay26shuffle' in file \
                and file.lower().endswith('.csv'):
            dam_files.append(video_directory + os.sep + file)
            dam_keys.append(file.split('DLC_resnet50_dam-single-animalMay26shuffle')[0])
    #print("Dam files:", dam_files)

    #given the video directory, find all files containing "DLC_dlcrnetms5_pup-nine-ptJun5shuffle" and ending in "UNPICKLED.csv"
    pup_files = []
    pup_keys= []
    for file in all_files:
        if 'DLC_dlcrnetms5_pup-nine-ptJun5shuffle' in file \
                and file.lower().endswith('unpickled.csv'):
            pup_files.append(video_directory + os.sep + file)
            pup_keys.append(file.split('DLC_dlcrnetms5_pup-nine-ptJun5shuffle')[0])
    #print("Pup files:", pup_files)

    #create paired_keys for files that have pup and dam pose estimation
    dam_set = set(dam_keys)
    pup_set = set(pup_keys)
    paired_keys = list(dam_set.intersection(pup_set))
    #print("Paired keys:", paired_keys)

    for key in paired_keys:
        print('Joining', key)
        for dam_file in dam_files:
            if video_directory + os.sep + key + 'DLC_resnet50_dam-single-animalMay26shuffle' in dam_file:
                dam_path = dam_file
                break

        dam_df = pd.read_csv(dam_path)
        dam_df.columns = dam_df.iloc[0] + '_' + dam_df.iloc[1]
        dam_df = dam_df.iloc[2:]
        dam_df = dam_df.rename(columns= {'bodyparts_coords': 'frame'})
        dam_df['frame'] =dam_df['frame'].astype(int)

        for pup_file in pup_files:
            if video_directory + os.sep + key + 'DLC_dlcrnetms5_pup-nine-ptJun5shuffle' in pup_file:
                pup_path = pup_file
                break

        pup_df = pd.read_csv(pup_path)

        merged_df = dam_df.merge(pup_df, on='frame', how='left')
        merged_df.loc[-2] = [column_name.replace('_x', '').replace('_y', '').replace('_likeihood','') for column_name in merged_df.columns]
        merged_df.loc[-1] = ['coords'] + ['x', 'y', 'likelihood'] * int((len(merged_df.columns) - 1) / 3)
        merged_df.index = merged_df.index + 2
        merged_df.sort_index(inplace=True)
        merged_df.columns = ['scorer'] + (['DLC_AMBER_dam_pup'] * (len(merged_df.columns) - 1))
        merged_df.iloc[0, 0] = 'bodyparts'

        try:
            os.makedirs(video_directory + os.sep + 'AMBER_joined_pose_estimation' + os.sep)
        except FileExistsError:
            pass
        out_path = video_directory + os.sep + 'AMBER_joined_pose_estimation' + os.sep + key + '.csv'
        merged_df.to_csv(out_path, index=False)
        #print(out_path)


def run_join_dam_pup():
    sys.exit(main(sys.argv))


if __name__ == '__main__':
    run_join_dam_pup()




