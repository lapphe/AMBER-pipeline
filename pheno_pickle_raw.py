# by Cem B and Hannah Lapp

import pickle
import sys

import numpy as np
import pandas as pd
import os
import ruamel.yaml

def show_help():
    print('PhenoPickleRaw!')
    print('Must specificy at the input_directory full path:')
    print(' -input_directory:[pickled files input folder full path]')
    print(' -input_file:[pickled file name without .pickle]')
    print(' -max_tracks:#')
    print(' -empty_cell_val:[]')
    print()
    print('Example: python PhenoPickleRaw.py -input_directory:/full/directory/path -max_tracks:11 -empty_cell_val:null')
    return


def unpickle_input(file_path, directory_path, max_tracks, empty_cell_val):
    """
    Converts the .pickle file containing pup detectionts, restructures its data, and exports it as a CSV file.
    Note: Although detected points are placed in columns contianing pup IDs, this placement is arbitrary and does not 
    necessarily mean that the dection belongs to that particular pup. For exmaple, the nose, eye, head, and back points
    assigned to 'pup1' may belong to the same pups or may belong to different pups. Likewise, point assignments are 
    not consistent across frames. Forexample, the nose point for pup1 in frame n and n+1 may belong to different 
    individuals.  

    Args:
        file_path (str): The path to the input pickle file.
        directory_path (str): The directory where the output CSV file will be saved.
        max_tracks (int): The maximum number of tracked items (e.g., pups) to extract from the pickle file.
        empty_cell_val: The value to fill empty cells with in the output CSV file.

    Returns:
        None

    This function reads a DataFrame from the specified pickle file, restructures the data to create a new DataFrame
    suitable for SimBA analysis, and exports it as a CSV file. It processes tracking data for multiple body parts and tracks
    (e.g., 'pup1_nose_x', 'pup2_eyes_y', etc.) and organizes it into rows based on the frame number. The CSV file is 
    saved with _UNPICLED.csv appended to the filename.

    Example:
        To unpickle a file 'video1.pickle' and save the output in the 'output_directory' with a maximum of 12 tracked pups
        and using a value of NA for empty cells:
        
        >>> unpickle_input('video1.pickle', 'output_directory', 12, 'NA')
    """
    print('   Unpickling', file_path)
    df = pd.read_pickle(file_path)
    print('   Read pickle file with', len(df), 'frames containing detections')

    body_parts = ['nose', 'eyes', 'ears', 'back1', 'back2', 'back3', 'back4', 'back5', 'back6']

    columns = ['frame']
    for pup in range(max_tracks):
        for part in body_parts:
            columns.append('pup' + str(pup+1) + '_' + part + '_x')
            columns.append('pup' + str(pup+1) + '_' + part + '_y')
            columns.append('pup' + str(pup+1) + '_' + part + '_likelihood')

    output = []

    counter = 0
    for k, v in df.items():
        counter += 1
        if not k.startswith('frame'):
            continue
        row = k[5:]

        data_row = dict.fromkeys(columns)
        data_row['frame'] = row

        # Read the coordinates and put them into data_row in the correct shape
        for bp, arr in enumerate(v['coordinates'][0]):
            for p, xy_coords in enumerate(arr):
                if p < max_tracks:
                    data_row['pup' + str(p+1) + '_' + body_parts[bp] + '_x'] = xy_coords[0]
                    data_row['pup' + str(p+1) + '_' + body_parts[bp] + '_y'] = xy_coords[1]

        # Read the confidences and put them into data_row
        for bp, arr in enumerate(v['confidence']):
            for p, pup_conf in enumerate(arr):
                if p < max_tracks:
                    data_row['pup' + str(p+1) + '_' + body_parts[bp] + '_likelihood'] = pup_conf[0]

        # for i in range(len(confidence_row)):
        output.append(data_row)

        if counter > 10000000:
            print('   Hit the maximum of 10 MILLION frames..')
            break

    output_file = pd.DataFrame.from_dict(output)
    output_file = output_file.set_index('frame')
    output_file.fillna(empty_cell_val, inplace=True)
    output_file = output_file.sort_values(by='frame')

    output_file_name = file_path.split('.')[-2].split(os.sep)[-1] + '_UNPICKLED.csv'

    output_path = directory_path + os.sep + output_file_name
    print('   Writing output to:', output_path)

    output_file.to_csv(output_path)
    print('   Done unpickling file!')


def main(args, max_tracks=12, empty_cell_val='NA', file_path=None, directory_path=None):
    """
    Executes the PhenoPickleRaw tool to unpickle and process raw tracking data from one or more pickle files.

    Args:
        max_tracks (int, optional): The maximum number of tracked pups to extract from the pickle files.
            Defaults to 12.
        empty_cell_val (str, optional): The value to fill empty cells with in the output CSV file. Defaults to 'NA'.
        file_path (str, optional): The path to an individual input pickle file (if specified). Defaults to None.
        directory_path (str, optional): The directory where input pickle files are located. Defaults to None.

    Returns:
        None

    This function is the entry point for the PhenoPickleRaw tool, which processes raw tracking data stored in pickle
    files. It accepts command-line arguments to specify input files, output format, and processing options.

    Args:
        - -input_directory:<path>: Specify the input directory containing pickle files created during pup pose estimation.
        - -input_file:<filename>: Specify an individual input pickle file.
        - -output:<output_path>: Specify the path for the output CSV file (must end with '.csv').
        - -max_tracks:<max_tracks>: Set the maximum number of tracked items to process.
        - -empty_cell_val:<value>: Set the value to fill empty cells in the output CSV.

    Example:
        To process tracking data from pickle files in the 'data' directory and save the output as 'output.csv' with a
        maximum of 5 tracked items and using 'NA' for empty cells:

        >>> main(['-input_directory:AMBER-pipeline/video_analysis', '-output:output.csv', '-max_tracks:5', '-empty_cell_val:NA'])

    Note:
        - If -input_file is specified, it will process the individual file.
        - If -input_directory is specified, it will search for all '.full.pickle' files in the directory.
        - The -output argument must end with '.csv'.
        - If no input files are found or specified, the function will print an error message and exit.

    """
    input_files = []
    file_name = None

    print()
    print()
    print('--- PhenoPickleRaw ---')

    if len(args) == 1:
        show_help()
        return

    if args[1] == '-h':
        show_help()
        return

    for arg in args:
        if arg.startswith('-input_directory:'):
            directory_path = arg[len('-input_directory:'):]
            directory_path = directory_path.replace("'", '')
            directory_path = directory_path.replace('"', '')
    if directory_path is None:
        print('Error! No input folder specified.')
        return

    for arg in args:
        if arg.startswith('-input_file:'):
            file_name = arg[len('-input_file:'):]
            input_files = directory_path + os.sep + file_name
    if file_name is None:
        print('Input file not specified, search directory.')
        files = os.listdir(directory_path)
        for file in files:
            if file.endswith('full.pickle'):
                input_files.append(directory_path + os.sep + file)
    print(len(input_files), 'full pickle files found.')

    for arg in args:
        if arg[0:7] == '-output':
            output_path = arg[8:]
            if output_path[-4:] != '.csv':
                print(output_path)
                print('Output does not end with .csv!', output_path[-4:])
                return

    for arg in args:
        if arg[0:12] == '-max_tracks':
            max_tracks = arg[13:]
            print('Max tracks:', max_tracks)

    for arg in args:
        if arg[0:17] == '-empty_cell_val':
            empty_cell_val = arg[18:]
            print('Empty cell val:', empty_cell_val)

    for file in input_files:
        if file.endswith('meta.pickle'):
            continue
        unpickle_input(file, directory_path, max_tracks=max_tracks, empty_cell_val=empty_cell_val)

    print('Done unpickling all files.')
    return


def run_phenopickle():
    sys.exit(main(sys.argv))


if __name__ == '__main__':
    run_phenopickle()
