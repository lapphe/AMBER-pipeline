#Hannah Lapp

#this script joins the dam and pup pose estimation files for the AMBER pipeline
import os
import sys
import pandas as pd

def main(argv):
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
        out_path = video_directory + os.sep + 'AMBER_joined_pose_estimation' + os.sep + key + '_dam_pup.csv'
        merged_df.to_csv(out_path, index=False)
        #print(out_path)


def run_join_dam_pup():
    sys.exit(main(sys.argv))


if __name__ == '__main__':
    run_join_dam_pup()




