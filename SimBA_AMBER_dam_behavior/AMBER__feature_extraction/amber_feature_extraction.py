from simba.misc_tools import (get_fn_ext,
                              SimbaTimer,
                              )
from simba.utils.printing import stdout_success
from simba.read_config_unit_tests import check_str
from simba.feature_extractors.unit_tests import read_video_info
from simba.rw_dfs import (read_df,
                          save_df)
import os
import pandas as pd
import numpy as np
from itertools import product
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from scipy.spatial import ConvexHull
from ellipse import LsqEllipse


def calculate_convex_hull(x, y, p, normalization_factor, threshold=0.2):
    normalization_factor = normalization_factor ** 2
    if len(x) != len(p):
        raise ValueError('Got x and p with different lengths')

    selected_points, selected_p = [], []
    for i in range(len(x)):
        if p[i] > threshold:
            selected_points.append([x[i], y[i]])
            selected_p.append(p[i])

    if len(selected_points) < 3:
        return 0
    else:
        return ConvexHull(selected_points).volume / normalization_factor


def calculate_weighted_avg(x, p=None, threshold=0.2):
    if p is not None and len(x) != len(p):
        raise ValueError('Got x and p with different lengths')

    selected_x, selected_p = [], []
    if p is not None:
        p = [0 if val is None else val for val in p]
        for i in range(len(x)):
            if p[i] > threshold:
                selected_x.append(x[i])
                selected_p.append(p[i])

    if len(selected_x) > 0:
        return np.ma.average(selected_x, weights=selected_p)
    else:
        return np.ma.average(x)


def count_high_p(p, threshold=0.2):
    return len([1 for val in p if val > threshold])


def get_ellipse_parameters(x, y, p, normalization_factor, threshold=0.2):
    selected_x, selected_y = [], []
    for i in range(len(p)):
        if p[i] > threshold:
            selected_x.append(x[i])
            selected_y.append(y[i])

    if len(selected_x) < 5:
        return 0, 0

    points = np.array(list(zip(selected_x, selected_y)))

    try:
        reg = LsqEllipse().fit(points)
        center, width, height, phi = reg.as_parameters()
        return np.real(width) / normalization_factor, np.real(height) / normalization_factor
    except IndexError:
        return 0.00001, 0.00001


class UserDefinedFeatureExtractor(FeatureExtractionMixin):
    """
    Class for featurizing data within SimBA project using user-defined body-parts in the pose-estimation data.
    Results are stored in the `project_folder/csv/features_extracted` directory of the SimBA project.

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format

    Notes
    ----------
    Feature extraction tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-5-extract-features>`__.

    Examples
    ----------
    >>> feature_extractor = UserDefinedFeatureExtractor(config_path='MyProjectConfig')
    >>> feature_extractor.extract_features()

    """
    def __init__(self, config_path: str):
        super().__init__(config_path=config_path)
        self.timer = SimbaTimer()
        self.timer.start_timer()
        print('Extracting features from {} file(s)...'.format(str(len(self.files_found))))
        self.extract_features()


    def extract_features(self):
        """
        Method to compute and save features to disk. Results are saved in the `project_folder/csv/features_extracted`
        directory of the SimBA project.

        Returns
        -------
        None
        """

        print('started extract_features')

        pup_threshold = 0.2
        dam_threshold = 0.4
        roll_windows_values = [1, 2, 5, 8, 0.5]  # values used to calculate rolling average across frames

        for file_cnt, file_path in enumerate(self.files_found):
            video_timer = SimbaTimer()
            video_timer.start_timer()
            print('Extracting features for video {}/{}...'.format(str(file_cnt+1), str(len(self.files_found))))
            _, file_name, _ = get_fn_ext(file_path)
            check_str('file name', file_name)
            video_settings, self.px_per_mm, fps = read_video_info(self.vid_info_df, file_name)
            pixels_per_mm = self.px_per_mm

            csv_df = read_df(file_path, self.file_type)

            # Body parts will be in two categories: dam and pups
            body_part_names = [x.replace('_x', '') for x in list(csv_df.columns) if '_x' in x]
            dam_body_part_names, pup_body_part_names = [], []
            for bp in body_part_names:
                if 'pup' in bp:
                    pup_body_part_names.append(bp)
                else:
                    dam_body_part_names.append(bp)

            # csv_df = csv_df.fillna(0)
            csv_df = csv_df.drop(csv_df.index[[0]])
            csv_df = csv_df.apply(pd.to_numeric)
            csv_df = csv_df.reset_index(drop=True)

            print('Calculating dam points, centroids, and convex hulls')

            # Collapse arm and side dam points
            csv_df['arm_x'] = np.where(csv_df['left_armpit_p'] > dam_threshold, csv_df['left_armpit_x'],
                                       csv_df['right_armpit_x'])
            csv_df['arm_y'] = np.where(csv_df['left_armpit_p'] > dam_threshold, csv_df['left_armpit_y'],
                                       csv_df['right_armpit_y'])
            csv_df['arm_p'] = np.where(csv_df['left_armpit_p'] > dam_threshold, csv_df['left_armpit_p'],
                                       csv_df['right_armpit_p'])

            csv_df['side_x'] = np.where(csv_df['left_ventrum_side_p'] > dam_threshold, csv_df['left_ventrum_side_x'],
                                        csv_df['right_ventrum_side_x'])
            csv_df['side_y'] = np.where(csv_df['left_ventrum_side_p'] > dam_threshold, csv_df['left_ventrum_side_y'],
                                        csv_df['right_ventrum_side_y'])
            csv_df['side_p'] = np.where(csv_df['left_ventrum_side_p'] > dam_threshold, csv_df['left_ventrum_side_p'],
                                        csv_df['right_ventrum_side_p'])

            # Calculate dam centroids and convex hulls
            csv_df['dam_centroid_x'] = csv_df.apply(lambda row: calculate_weighted_avg(
                [row[str(column) + '_x'] for column in dam_body_part_names],
                [row[str(column) + '_p'] for column in dam_body_part_names],
                threshold=dam_threshold), axis=1)
            csv_df['dam_centroid_y'] = csv_df.apply(lambda row: calculate_weighted_avg(
                [row[str(column) + '_y'] for column in dam_body_part_names],
                [row[str(column) + '_p'] for column in dam_body_part_names],
                threshold=dam_threshold), axis=1)

            csv_df['dam_convex_hull'] = csv_df.apply(lambda row: calculate_convex_hull(
                [row[p + '_x'] for p in dam_body_part_names],
                [row[p + '_y'] for p in dam_body_part_names],
                [row[p + '_p'] for p in dam_body_part_names], pixels_per_mm), axis=1)

            centroid_exclusion_parts = ['tail_base', 'left_palm', 'right_palm', 'left_ankle', 'right_ankle']
            dam_core_parts = [p for p in dam_body_part_names if p not in centroid_exclusion_parts]
            csv_df['dam_core_convex_hull'] = csv_df.apply(lambda row: calculate_convex_hull(
                [row[p + '_x'] for p in dam_core_parts],
                [row[p + '_y'] for p in dam_core_parts],
                [row[p + '_p'] for p in dam_core_parts], pixels_per_mm), axis=1)

            dam_head_parts = ['dam_nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'top_head_dam']
            csv_df['head_centroid_x'] = csv_df.apply(lambda row: calculate_weighted_avg(
                [row[str(column) + '_x'] for column in dam_head_parts],
                [row[str(column) + '_p'] for column in dam_head_parts],
                threshold=dam_threshold), axis=1)
            csv_df['head_centroid_y'] = csv_df.apply(lambda row: calculate_weighted_avg(
                [row[str(column) + '_y'] for column in dam_head_parts],
                [row[str(column) + '_p'] for column in dam_head_parts],
                threshold=dam_threshold), axis=1)

            csv_df['head_convex_hull'] = csv_df.apply(lambda row: calculate_convex_hull(
                [row[p + '_x'] for p in dam_head_parts],
                [row[p + '_y'] for p in dam_head_parts],
                [row[p + '_p'] for p in dam_head_parts], pixels_per_mm), axis=1)

            dam_exclusion_parts = ['left_ankle', 'right_ankle', 'tail_base', 'left_palm', 'right_palm', 'btwn_arms',
                                   'btwn_legs', 'center_ventrum']
            dam_center_parts = [p for p in dam_body_part_names if p not in dam_exclusion_parts]
            csv_df['dam_core2_convex_hull'] = csv_df.apply(lambda row: calculate_convex_hull(
                [row[p + '_x'] for p in dam_center_parts],
                [row[p + '_y'] for p in dam_center_parts],
                [row[p + '_p'] for p in dam_center_parts], pixels_per_mm), axis=1)

            # Calculate the center of all pup points
            print('Calculating pup centroids')

            pup_columns_x, pup_columns_y, pup_columns_p = [], [], []
            for bp in pup_body_part_names:
                pup_columns_x.append(str(bp) + '_x')
                pup_columns_y.append(str(bp) + '_y')
                pup_columns_p.append(str(bp) + '_p')

            csv_df['pups_centroid_x'] = csv_df.apply(lambda row: calculate_weighted_avg(
                [row[p] for p in pup_columns_x],
                [row[p] for p in pup_columns_p],
                threshold=pup_threshold), axis=1)
            csv_df['pups_centroid_y'] = csv_df.apply(lambda row: calculate_weighted_avg(
                [row[p] for p in pup_columns_y],
                [row[p] for p in pup_columns_p],
                threshold=pup_threshold), axis=1)

            print('Calculating pup convex hull')
            csv_df['pups_convex_hull'] = csv_df.apply(lambda row: calculate_convex_hull(
                [row[p] for p in pup_columns_x],
                [row[p] for p in pup_columns_y],
                [row[p] for p in pup_columns_p], pixels_per_mm), axis=1)

            print('Calculating high probably body part counts')
            # csv_df['pup_avg_p'] = csv_df.apply(lambda row: calculate_weighted_avg([row[p] for p in pup_columns_p],
            #                                                                       threshold=pup_threshold), axis=1)
            csv_df['pup_avg_p'] = csv_df[[p for p in pup_columns_p]].mean(axis=1)

            csv_df['high_p_pup_bp'] = csv_df.apply(lambda row: count_high_p([row[p] for p in pup_columns_p]), axis=1)

            csv_df['high_p_dam_bp'] = csv_df.apply(lambda row: count_high_p([row[p + '_p'] for p in dam_body_part_names]
                                                                            ), axis=1)

            # Calculate movements
            print('Calculating movements')
            movement_columns = dam_body_part_names

            # Create a shifted dataframe and combine to use fo rmovement calculations
            csv_df_shifted = csv_df.shift(periods=1)
            csv_df_shifted.columns = [i + '_shifted' for i in csv_df.columns.values.tolist()]
            csv_df_combined = pd.concat([csv_df, csv_df_shifted], axis=1, join='inner')
            csv_df_combined = csv_df_combined.fillna(0)
            csv_df_combined = csv_df_combined.reset_index(drop=True)

            for bp in movement_columns:
                column_name = bp + '_movement'
                x1, y1 = (bp + '_x', bp + '_y')
                x2, y2 = (bp + '_x_shifted', bp + '_y_shifted')
                csv_df[column_name] = (np.sqrt((csv_df_combined[x1] - csv_df_combined[x2]) ** 2 +
                                               (csv_df_combined[y1] - csv_df_combined[y2]) ** 2)) / pixels_per_mm
                csv_df.at[0, column_name] = np.average(csv_df[column_name].iloc[1:10])

            back_point_movements = ['back_2_movement', 'back_3_movement',
                                    'back_4_movement', 'back_5_movement',
                                    'back_6_movement', 'back_7_movement',
                                    'back_8_movement', 'back_9_movement',
                                    'back_10_movement']
            back_point_movements_p = ['back_2_p', 'back_3_p',
                                      'back_4_p', 'back_5_p',
                                      'back_6_p', 'back_7_p',
                                      'back_8_p', 'back_9_p',
                                      'back_10_p']

            csv_df['back_avg_movement'] = csv_df.apply(lambda row: calculate_weighted_avg(
                [row[c] for c in back_point_movements],
                [row[c] for c in back_point_movements_p],
                threshold=dam_threshold), axis=1)

            head_point_movements = ['dam_nose_movement', 'right_eye_movement', 'left_eye_movement', 'left_ear_movement',
                                    'right_ear_movement']
            head_point_movements_p = ['dam_nose_p', 'right_eye_p', 'left_eye_p', 'left_ear_p',
                                      'right_ear_p']

            csv_df['head_avg_movement'] = csv_df.apply(lambda row: calculate_weighted_avg(
                [row[d] for d in head_point_movements],
                [row[d] for d in head_point_movements_p],
                threshold=dam_threshold), axis=1)

            csv_df['head_max_movement'] = np.ma.max([csv_df['dam_nose_movement'],
                                                     csv_df['right_eye_movement'],
                                                     csv_df['left_eye_movement'],
                                                     csv_df['left_ear_movement'],
                                                     csv_df['right_ear_movement']], axis=0)

            csv_df['ventrum_side_movement'] = csv_df.apply(lambda row: calculate_weighted_avg(
                [row['left_ventrum_side_movement'], row['right_ventrum_side_movement']],
                [row['left_ventrum_side_p'], row['right_ventrum_side_p']],
                threshold=dam_threshold), axis=1)

            csv_df['leg_front_movement'] = csv_df.apply(lambda row: calculate_weighted_avg(
                [row['left_leg_front_movement'], row['right_leg_front_movement']],
                [row['left_leg_front_p'], row['right_leg_front_p']],
                threshold=dam_threshold), axis=1)

            csv_df['leg_behind_movement'] = csv_df.apply(lambda row: calculate_weighted_avg(
                [row['left_leg_behind_movement'], row['right_leg_behind_movement']],
                [row['left_leg_behind_p'], row['right_leg_behind_p']],
                threshold=dam_threshold), axis=1)

            csv_df['wrist_movement'] = csv_df.apply(lambda row: calculate_weighted_avg(
                [row['left_wrist_movement'], row['right_wrist_movement']],
                [row['left_wrist_p'], row['right_wrist_p']],
                threshold=dam_threshold), axis=1)

            csv_df['armpit_movement'] = csv_df.apply(lambda row: calculate_weighted_avg(
                [row['left_armpit_movement'], row['right_armpit_movement']],
                [row['left_armpit_p'], row['right_armpit_p']],
                threshold=dam_threshold), axis=1)

            csv_df['shoulder_movement'] = csv_df.apply(lambda row: calculate_weighted_avg(
                [row['left_shoulder_movement'], row['right_shoulder_movement']],
                [row['left_shoulder_p'], row['right_shoulder_p']],
                threshold=dam_threshold), axis=1)

            # # Distance calculations
            print('Calculating distances')
            csv_df['dam_pup_distance'] = np.sqrt((csv_df['dam_centroid_x'] - csv_df['pups_centroid_x']) ** 2 +
                                                 (csv_df['dam_centroid_y'] - csv_df[
                                                     'pups_centroid_y']) ** 2) / pixels_per_mm
            csv_df['head_pup_distance'] = np.sqrt((csv_df['head_centroid_x'] - csv_df['pups_centroid_x']) ** 2 +
                                                  (csv_df['head_centroid_y'] - csv_df[
                                                      'pups_centroid_y']) ** 2) / pixels_per_mm

            csv_df['back_length'] = np.sqrt((csv_df['back_2_x'] - csv_df['back_10_x']) ** 2
                                            + (csv_df['back_2_y'] - csv_df['back_10_y']) ** 2) / pixels_per_mm

            csv_df['nose_back10_length'] = np.sqrt((csv_df['dam_nose_x'] - csv_df['back_10_x']) ** 2
                                                   + (csv_df['dam_nose_y'] - csv_df['back_10_y']) ** 2) / pixels_per_mm

            csv_df['back1_back10_length'] = np.sqrt((csv_df['back_1_center_x'] - csv_df['back_10_x']) ** 2
                                                    + (csv_df['back_1_center_y'] - csv_df[
                'back_10_y']) ** 2) / pixels_per_mm

            csv_df['nose_back2_length'] = np.sqrt((csv_df['dam_nose_x'] - csv_df['back_2_x']) ** 2
                                                  + (csv_df['dam_nose_y'] - csv_df['back_2_y']) ** 2) / pixels_per_mm

            csv_df['left_wrist_nose_length'] = np.sqrt((csv_df['left_wrist_x'] - csv_df['dam_nose_x']) ** 2 +
                                                       (csv_df['left_wrist_y'] - csv_df[
                                                           'dam_nose_y']) ** 2) / pixels_per_mm
            csv_df['right_wrist_nose_length'] = np.sqrt((csv_df['right_wrist_x'] - csv_df['dam_nose_x']) ** 2 +
                                                        (csv_df['right_wrist_y'] - csv_df[
                                                            'dam_nose_y']) ** 2) / pixels_per_mm
            csv_df['wrist_nose_length'] = csv_df.apply(lambda row: calculate_weighted_avg(
                [row['left_wrist_nose_length'], row['right_wrist_nose_length']],
                [row['left_wrist_p'], row['right_wrist_p']],
                threshold=dam_threshold), axis=1)
            csv_df.drop(inplace=True, columns=['left_wrist_nose_length', 'right_wrist_nose_length'])

            csv_df['avg_dam_bp_p'] = np.ma.average([csv_df['dam_nose_p'],
                                                    csv_df['left_eye_p'],
                                                    csv_df['right_eye_p'],
                                                    csv_df['left_ear_p'],
                                                    csv_df['right_ear_p'],
                                                    csv_df['left_shoulder_p'],
                                                    csv_df['right_shoulder_p'],
                                                    csv_df['arm_p'],
                                                    csv_df['side_p']], axis=0)

            csv_df['sum_probabilities'] = csv_df[[p + '_p' for p in body_part_names]].sum(axis=1)

            print('Calculating fields for dam back curve')
            back_points_x = ['back_1_center_x', 'back_2_x', 'back_3_x', 'back_4_x', 'back_5_x', 'back_6_x', 'back_7_x',
                             'back_8_x', 'back_9_x', 'back_10_x']
            back_points_y = ['back_1_center_y', 'back_2_y', 'back_3_y', 'back_4_y', 'back_5_y', 'back_6_y', 'back_7_y',
                             'back_8_y', 'back_9_y', 'back_10_y']
            back_points_p = ['back_1_center_p', 'back_2_p', 'back_3_p', 'back_4_p', 'back_5_p', 'back_6_p', 'back_7_p',
                             'back_8_p', 'back_9_p', 'back_10_p']
            csv_df['back_ellipse_width'] = csv_df.apply(lambda row: get_ellipse_parameters(
                [row[p] for p in back_points_x],
                [row[p] for p in back_points_y],
                [row[p] for p in back_points_p],
                pixels_per_mm,
                threshold=dam_threshold)[0], axis=1)
            csv_df['back_ellipse_height'] = csv_df.apply(lambda row: get_ellipse_parameters(
                [row[p] for p in back_points_x],
                [row[p] for p in back_points_y],
                [row[p] for p in back_points_p],
                pixels_per_mm,
                threshold=dam_threshold)[1], axis=1)
            csv_df['back_ellipse_ratio'] = csv_df.apply(
                lambda row: row['back_ellipse_width'] / row['back_ellipse_height']
                if row['back_ellipse_height'] > 0 else 0, axis=1)

            # # Moving averages
            print('Calculating rolling average')

            roll_windows = []
            for j in range(len(roll_windows_values)):
                roll_windows.append(int(fps / roll_windows_values[j]))

            csv_df['head_avg_movement_roll_mean_1s'] = csv_df['head_avg_movement'].rolling(roll_windows[0],
                                                                                           min_periods=1).mean()
            csv_df['head_avg_movement_roll_mean_1ds'] = csv_df['head_avg_movement'].rolling(roll_windows[2],
                                                                                            min_periods=1).mean()
            csv_df['head_avg_movement_roll_mean_2s'] = csv_df['head_avg_movement'].rolling(roll_windows[4],
                                                                                           min_periods=1).mean()

            csv_df['head_max_movement_roll_mean_1s'] = csv_df['head_max_movement'].rolling(roll_windows[0],
                                                                                           min_periods=1).mean()
            csv_df['head_max_movement_roll_mean_1ds'] = csv_df['head_max_movement'].rolling(roll_windows[2],
                                                                                            min_periods=1).mean()
            csv_df['head_max_movement_roll_mean_2s'] = csv_df['head_max_movement'].rolling(roll_windows[4],
                                                                                           min_periods=1).mean()

            csv_df['back_avg_movement_roll_mean_1s'] = csv_df['back_avg_movement'].rolling(roll_windows[0],
                                                                                           min_periods=1).mean()
            csv_df['back_avg_movement_roll_mean_1ds'] = csv_df['back_avg_movement'].rolling(roll_windows[2],
                                                                                            min_periods=1).mean()
            csv_df['back_avg_movement_roll_mean_2s'] = csv_df['back_avg_movement'].rolling(roll_windows[4],
                                                                                           min_periods=1).mean()

            csv_df['head_back_rel_roll_mean_1s'] = csv_df['head_avg_movement_roll_mean_1s'] / (
                    csv_df['head_avg_movement_roll_mean_1s'] +
                    csv_df['back_avg_movement_roll_mean_1s'])
            csv_df['head_back_rel_roll_mean_1ds'] = csv_df['head_avg_movement_roll_mean_1ds'] / (
                    csv_df['head_avg_movement_roll_mean_1ds'] +
                    csv_df['back_avg_movement_roll_mean_1ds'])
            csv_df['head_back_rel_roll_mean_2s'] = csv_df['head_avg_movement_roll_mean_2s'] / (
                    csv_df['head_avg_movement_roll_mean_2s'] +
                    csv_df['back_avg_movement_roll_mean_2s'])

            csv_df['pups_convex_hull_roll_mean_1s'] = csv_df['pups_convex_hull'].rolling(roll_windows[0],
                                                                                         min_periods=1).mean()
            csv_df['pups_convex_hull_roll_mean_1ds'] = csv_df['pups_convex_hull'].rolling(roll_windows[2],
                                                                                          min_periods=1).mean()
            csv_df['pups_convex_hull_roll_mean_2s'] = csv_df['pups_convex_hull'].rolling(roll_windows[4],
                                                                                         min_periods=1).mean()

            csv_df['dam_pup_distance_roll_mean_1s'] = csv_df['dam_pup_distance'].rolling(roll_windows[0],
                                                                                         min_periods=1).mean()
            csv_df['dam_pup_distance_roll_mean_1ds'] = csv_df['dam_pup_distance'].rolling(roll_windows[2],
                                                                                          min_periods=1).mean()
            csv_df['dam_pup_distance_roll_mean_2s'] = csv_df['dam_pup_distance'].rolling(roll_windows[4],
                                                                                         min_periods=1).mean()

            csv_df['back_ellipse_ratio_roll_mean_1s'] = csv_df['back_ellipse_ratio'].rolling(roll_windows[0],
                                                                                             min_periods=1).mean()
            csv_df['back_ellipse_ratio_roll_mean_1ds'] = csv_df['back_ellipse_ratio'].rolling(roll_windows[2],
                                                                                              min_periods=1).mean()
            csv_df['back_ellipse_ratio_roll_mean_2s'] = csv_df['back_ellipse_ratio'].rolling(roll_windows[4],
                                                                                             min_periods=1).mean()

            csv_df['dam_nose_movement_roll_mean_1s'] = csv_df['dam_nose_movement'].rolling(roll_windows[0],
                                                                                           min_periods=1).mean()
            csv_df['dam_nose_movement_roll_mean_1ds'] = csv_df['dam_nose_movement'].rolling(roll_windows[2],
                                                                                            min_periods=1).mean()
            csv_df['dam_nose_movement_roll_mean_2s'] = csv_df['dam_nose_movement'].rolling(roll_windows[4],
                                                                                           min_periods=1).mean()

            csv_df['left_eye_movement_roll_mean_1s'] = csv_df['left_eye_movement'].rolling(roll_windows[0],
                                                                                           min_periods=1).mean()
            csv_df['left_eye_movement_roll_mean_1ds'] = csv_df['left_eye_movement'].rolling(roll_windows[2],
                                                                                            min_periods=1).mean()
            csv_df['left_eye_movement_roll_mean_2s'] = csv_df['left_eye_movement'].rolling(roll_windows[4],
                                                                                           min_periods=1).mean()

            csv_df['right_eye_movement_roll_mean_1s'] = csv_df['right_eye_movement'].rolling(roll_windows[0],
                                                                                             min_periods=1).mean()
            csv_df['right_eye_movement_roll_mean_1ds'] = csv_df['right_eye_movement'].rolling(roll_windows[2],
                                                                                              min_periods=1).mean()
            csv_df['right_eye_movement_roll_mean_2s'] = csv_df['right_eye_movement'].rolling(roll_windows[4],
                                                                                             min_periods=1).mean()

            csv_df['left_ear_movement_roll_mean_1s'] = csv_df['left_ear_movement'].rolling(roll_windows[0],
                                                                                           min_periods=1).mean()
            csv_df['left_ear_movement_roll_mean_1ds'] = csv_df['left_ear_movement'].rolling(roll_windows[2],
                                                                                            min_periods=1).mean()
            csv_df['left_ear_movement_roll_mean_2s'] = csv_df['left_ear_movement'].rolling(roll_windows[4],
                                                                                           min_periods=1).mean()

            csv_df['right_ear_movement_roll_mean_1s'] = csv_df['right_ear_movement'].rolling(roll_windows[0],
                                                                                             min_periods=1).mean()
            csv_df['right_ear_movement_roll_mean_1ds'] = csv_df['right_ear_movement'].rolling(roll_windows[2],
                                                                                              min_periods=1).mean()
            csv_df['right_ear_movement_roll_mean_2s'] = csv_df['right_ear_movement'].rolling(roll_windows[4],
                                                                                             min_periods=1).mean()

            csv_df['ventrum_side_movement_roll_mean_1s'] = csv_df['ventrum_side_movement'].rolling(roll_windows[0],
                                                                                                   min_periods=1).mean()
            csv_df['ventrum_side_movement_roll_mean_1ds'] = csv_df['ventrum_side_movement'].rolling(roll_windows[2],
                                                                                                    min_periods=1).mean()
            csv_df['ventrum_side_movement_roll_mean_2s'] = csv_df['ventrum_side_movement'].rolling(roll_windows[4],
                                                                                                   min_periods=1).mean()

            csv_df['leg_front_movement_roll_mean_1s'] = csv_df['leg_front_movement'].rolling(roll_windows[0],
                                                                                             min_periods=1).mean()
            csv_df['leg_front_movement_roll_mean_1ds'] = csv_df['leg_front_movement'].rolling(roll_windows[2],
                                                                                              min_periods=1).mean()
            csv_df['leg_front_movement_roll_mean_2s'] = csv_df['leg_front_movement'].rolling(roll_windows[4],
                                                                                             min_periods=1).mean()

            csv_df['leg_behind_movement_roll_mean_1s'] = csv_df['leg_behind_movement'].rolling(roll_windows[0],
                                                                                               min_periods=1).mean()
            csv_df['leg_behind_movement_roll_mean_1ds'] = csv_df['leg_behind_movement'].rolling(roll_windows[2],
                                                                                                min_periods=1).mean()
            csv_df['leg_behind_movement_roll_mean_2s'] = csv_df['leg_behind_movement'].rolling(roll_windows[4],
                                                                                               min_periods=1).mean()

            csv_df['wrist_movement_roll_mean_1s'] = csv_df['wrist_movement'].rolling(roll_windows[0],
                                                                                     min_periods=1).mean()
            csv_df['wrist_movement_roll_mean_1ds'] = csv_df['wrist_movement'].rolling(roll_windows[2],
                                                                                      min_periods=1).mean()
            csv_df['wrist_movement_roll_mean_2s'] = csv_df['wrist_movement'].rolling(roll_windows[4],
                                                                                     min_periods=1).mean()

            csv_df['armpit_movement_roll_mean_1s'] = csv_df['armpit_movement'].rolling(roll_windows[0],
                                                                                       min_periods=1).mean()
            csv_df['armpit_movement_roll_mean_1ds'] = csv_df['armpit_movement'].rolling(roll_windows[2],
                                                                                        min_periods=1).mean()
            csv_df['armpit_movement_roll_mean_2s'] = csv_df['armpit_movement'].rolling(roll_windows[4],
                                                                                       min_periods=1).mean()

            csv_df['shoulder_movement_roll_mean_1s'] = csv_df['shoulder_movement'].rolling(roll_windows[0],
                                                                                           min_periods=1).mean()
            csv_df['shoulder_movement_roll_mean_1ds'] = csv_df['shoulder_movement'].rolling(roll_windows[2],
                                                                                            min_periods=1).mean()
            csv_df['shoulder_movement_roll_mean_2s'] = csv_df['shoulder_movement'].rolling(roll_windows[4],
                                                                                           min_periods=1).mean()

            csv_df['dam_convex_hull_roll_mean_1s'] = csv_df['dam_convex_hull'].rolling(roll_windows[0],
                                                                                       min_periods=1).mean()
            csv_df['dam_convex_hull_roll_mean_1ds'] = csv_df['dam_convex_hull'].rolling(roll_windows[2],
                                                                                        min_periods=1).mean()
            csv_df['dam_convex_hull_roll_mean_2s'] = csv_df['dam_convex_hull'].rolling(roll_windows[4],
                                                                                       min_periods=1).mean()

            csv_df['head_convex_hull_roll_mean_1s'] = csv_df['head_convex_hull'].rolling(roll_windows[0],
                                                                                         min_periods=1).mean()
            csv_df['head_convex_hull_roll_mean_1ds'] = csv_df['head_convex_hull'].rolling(roll_windows[2],
                                                                                          min_periods=1).mean()
            csv_df['head_convex_hull_roll_mean_2s'] = csv_df['head_convex_hull'].rolling(roll_windows[4],
                                                                                         min_periods=1).mean()

            csv_df['high_p_pup_bp_roll_mean_1s'] = csv_df['high_p_pup_bp'].rolling(roll_windows[0],
                                                                                   min_periods=1).mean()
            csv_df['high_p_pup_bp_roll_mean_1ds'] = csv_df['high_p_pup_bp'].rolling(roll_windows[2],
                                                                                    min_periods=1).mean()
            csv_df['high_p_pup_bp_roll_mean_2s'] = csv_df['high_p_pup_bp'].rolling(roll_windows[4],
                                                                                   min_periods=1).mean()

            csv_df['high_p_dam_bp_roll_mean_1s'] = csv_df['high_p_dam_bp'].rolling(roll_windows[0],
                                                                                   min_periods=1).mean()
            csv_df['high_p_dam_bp_roll_mean_1ds'] = csv_df['high_p_dam_bp'].rolling(roll_windows[2],
                                                                                    min_periods=1).mean()
            csv_df['high_p_dam_bp_roll_mean_2s'] = csv_df['high_p_dam_bp'].rolling(roll_windows[4],
                                                                                   min_periods=1).mean()

            csv_df['nose_back10_length_roll_mean_1s'] = csv_df['nose_back10_length'].rolling(roll_windows[0],
                                                                                             min_periods=1).mean()
            csv_df['nose_back10_length_roll_mean_1ds'] = csv_df['nose_back10_length'].rolling(roll_windows[2],
                                                                                              min_periods=1).mean()
            csv_df['nose_back10_length_roll_mean_2s'] = csv_df['nose_back10_length'].rolling(roll_windows[4],
                                                                                             min_periods=1).mean()

            # # Rolling sums
            print('Calculating rolling window sums')

            csv_df['head_avg_movement_roll_sum_1s'] = csv_df['head_avg_movement'].rolling(roll_windows[0],
                                                                                          min_periods=1).sum()
            csv_df['head_avg_movement_roll_sum_1ds'] = csv_df['head_avg_movement'].rolling(roll_windows[2],
                                                                                           min_periods=1).sum()
            csv_df['head_avg_movement_roll_sum_2s'] = csv_df['head_avg_movement'].rolling(roll_windows[4],
                                                                                          min_periods=1).sum()

            csv_df['head_max_movement_roll_sum_1s'] = csv_df['head_max_movement'].rolling(roll_windows[0],
                                                                                          min_periods=1).sum()
            csv_df['head_max_movement_roll_sum_1ds'] = csv_df['head_max_movement'].rolling(roll_windows[2],
                                                                                           min_periods=1).sum()
            csv_df['head_max_movement_roll_sum_2s'] = csv_df['head_max_movement'].rolling(roll_windows[4],
                                                                                          min_periods=1).sum()

            csv_df['back_avg_movement_roll_sum_1s'] = csv_df['back_avg_movement'].rolling(roll_windows[0],
                                                                                          min_periods=1).sum()
            csv_df['back_avg_movement_roll_sum_1ds'] = csv_df['back_avg_movement'].rolling(roll_windows[2],
                                                                                           min_periods=1).sum()
            csv_df['back_avg_movement_roll_sum_2s'] = csv_df['back_avg_movement'].rolling(roll_windows[4],
                                                                                          min_periods=1).sum()

            csv_df['head_back_rel_roll_sum_1s'] = csv_df['head_avg_movement_roll_sum_1s'] / (
                    csv_df['head_avg_movement_roll_sum_1s'] +
                    csv_df['back_avg_movement_roll_sum_1s'])
            csv_df['head_back_rel_roll_sum_1ds'] = csv_df['head_avg_movement_roll_sum_1ds'] / (
                    csv_df['head_avg_movement_roll_sum_1ds'] +
                    csv_df['back_avg_movement_roll_sum_1ds'])
            csv_df['head_back_rel_roll_sum_2s'] = csv_df['head_avg_movement_roll_sum_2s'] / (
                    csv_df['head_avg_movement_roll_sum_2s'] +
                    csv_df['back_avg_movement_roll_sum_2s'])

            csv_df['pups_convex_hull_roll_sum_1s'] = csv_df['pups_convex_hull'].rolling(roll_windows[0],
                                                                                        min_periods=1).sum()
            csv_df['pups_convex_hull_roll_sum_1ds'] = csv_df['pups_convex_hull'].rolling(roll_windows[2],
                                                                                         min_periods=1).sum()
            csv_df['pups_convex_hull_roll_sum_2s'] = csv_df['pups_convex_hull'].rolling(roll_windows[4],
                                                                                        min_periods=1).sum()

            csv_df['dam_pup_distance_roll_sum_1s'] = csv_df['dam_pup_distance'].rolling(roll_windows[0],
                                                                                        min_periods=1).sum()
            csv_df['dam_pup_distance_roll_sum_1ds'] = csv_df['dam_pup_distance'].rolling(roll_windows[2],
                                                                                         min_periods=1).sum()
            csv_df['dam_pup_distance_roll_sum_2s'] = csv_df['dam_pup_distance'].rolling(roll_windows[4],
                                                                                        min_periods=1).sum()

            csv_df['back_ellipse_ratio_roll_sum_1s'] = csv_df['back_ellipse_ratio'].rolling(roll_windows[0],
                                                                                            min_periods=1).sum()
            csv_df['back_ellipse_ratio_roll_sum_1ds'] = csv_df['back_ellipse_ratio'].rolling(roll_windows[2],
                                                                                             min_periods=1).sum()
            csv_df['back_ellipse_ratio_roll_sum_2s'] = csv_df['back_ellipse_ratio'].rolling(roll_windows[4],
                                                                                            min_periods=1).sum()

            csv_df['dam_nose_movement_roll_sum_1s'] = csv_df['dam_nose_movement'].rolling(roll_windows[0],
                                                                                          min_periods=1).sum()
            csv_df['dam_nose_movement_roll_sum_1ds'] = csv_df['dam_nose_movement'].rolling(roll_windows[2],
                                                                                           min_periods=1).sum()
            csv_df['dam_nose_movement_roll_sum_2s'] = csv_df['dam_nose_movement'].rolling(roll_windows[4],
                                                                                          min_periods=1).sum()

            csv_df['left_eye_movement_roll_sum_1s'] = csv_df['left_eye_movement'].rolling(roll_windows[0],
                                                                                          min_periods=1).sum()
            csv_df['left_eye_movement_roll_sum_1ds'] = csv_df['left_eye_movement'].rolling(roll_windows[2],
                                                                                           min_periods=1).sum()
            csv_df['left_eye_movement_roll_sum_2s'] = csv_df['left_eye_movement'].rolling(roll_windows[4],
                                                                                          min_periods=1).sum()

            csv_df['right_eye_movement_roll_sum_1s'] = csv_df['right_eye_movement'].rolling(roll_windows[0],
                                                                                            min_periods=1).sum()
            csv_df['right_eye_movement_roll_sum_1ds'] = csv_df['right_eye_movement'].rolling(roll_windows[2],
                                                                                             min_periods=1).sum()
            csv_df['right_eye_movement_roll_sum_2s'] = csv_df['right_eye_movement'].rolling(roll_windows[4],
                                                                                            min_periods=1).sum()

            csv_df['left_ear_movement_roll_sum_1s'] = csv_df['left_ear_movement'].rolling(roll_windows[0],
                                                                                          min_periods=1).sum()
            csv_df['left_ear_movement_roll_sum_1ds'] = csv_df['left_ear_movement'].rolling(roll_windows[2],
                                                                                           min_periods=1).sum()
            csv_df['left_ear_movement_roll_sum_2s'] = csv_df['left_ear_movement'].rolling(roll_windows[4],
                                                                                          min_periods=1).sum()

            csv_df['right_ear_movement_roll_sum_1s'] = csv_df['right_ear_movement'].rolling(roll_windows[0],
                                                                                            min_periods=1).sum()
            csv_df['right_ear_movement_roll_sum_1ds'] = csv_df['right_ear_movement'].rolling(roll_windows[2],
                                                                                             min_periods=1).sum()
            csv_df['right_ear_movement_roll_sum_2s'] = csv_df['right_ear_movement'].rolling(roll_windows[4],
                                                                                            min_periods=1).sum()

            csv_df['ventrum_side_movement_roll_sum_1s'] = csv_df['ventrum_side_movement'].rolling(roll_windows[0],
                                                                                                  min_periods=1).sum()
            csv_df['ventrum_side_movement_roll_sum_1ds'] = csv_df['ventrum_side_movement'].rolling(roll_windows[2],
                                                                                                   min_periods=1).sum()
            csv_df['ventrum_side_movement_roll_sum_2s'] = csv_df['ventrum_side_movement'].rolling(roll_windows[4],
                                                                                                  min_periods=1).sum()

            csv_df['leg_front_movement_roll_sum_1s'] = csv_df['leg_front_movement'].rolling(roll_windows[0],
                                                                                            min_periods=1).sum()
            csv_df['leg_front_movement_roll_sum_1ds'] = csv_df['leg_front_movement'].rolling(roll_windows[2],
                                                                                             min_periods=1).sum()
            csv_df['leg_front_movement_roll_sum_2s'] = csv_df['leg_front_movement'].rolling(roll_windows[4],
                                                                                            min_periods=1).sum()

            csv_df['leg_behind_movement_roll_sum_1s'] = csv_df['leg_behind_movement'].rolling(roll_windows[0],
                                                                                              min_periods=1).sum()
            csv_df['leg_behind_movement_roll_sum_1ds'] = csv_df['leg_behind_movement'].rolling(roll_windows[2],
                                                                                               min_periods=1).sum()
            csv_df['leg_behind_movement_roll_sum_2s'] = csv_df['leg_behind_movement'].rolling(roll_windows[4],
                                                                                              min_periods=1).sum()

            csv_df['wrist_movement_roll_sum_1s'] = csv_df['wrist_movement'].rolling(roll_windows[0],
                                                                                    min_periods=1).sum()
            csv_df['wrist_movement_roll_sum_1ds'] = csv_df['wrist_movement'].rolling(roll_windows[2],
                                                                                     min_periods=1).sum()
            csv_df['wrist_movement_roll_sum_2s'] = csv_df['wrist_movement'].rolling(roll_windows[4],
                                                                                    min_periods=1).sum()

            csv_df['armpit_movement_roll_sum_1s'] = csv_df['armpit_movement'].rolling(roll_windows[0],
                                                                                      min_periods=1).sum()
            csv_df['armpit_movement_roll_sum_1ds'] = csv_df['armpit_movement'].rolling(roll_windows[2],
                                                                                       min_periods=1).sum()
            csv_df['armpit_movement_roll_sum_2s'] = csv_df['armpit_movement'].rolling(roll_windows[4],
                                                                                      min_periods=1).sum()

            csv_df['shoulder_movement_roll_sum_1s'] = csv_df['shoulder_movement'].rolling(roll_windows[0],
                                                                                          min_periods=1).sum()
            csv_df['shoulder_movement_roll_sum_1ds'] = csv_df['shoulder_movement'].rolling(roll_windows[2],
                                                                                           min_periods=1).sum()
            csv_df['shoulder_movement_roll_sum_2s'] = csv_df['shoulder_movement'].rolling(roll_windows[4],
                                                                                          min_periods=1).sum()

            csv_df['dam_convex_hull_roll_sum_1s'] = csv_df['dam_convex_hull'].rolling(roll_windows[0],
                                                                                      min_periods=1).sum()
            csv_df['dam_convex_hull_roll_sum_1ds'] = csv_df['dam_convex_hull'].rolling(roll_windows[2],
                                                                                       min_periods=1).sum()
            csv_df['dam_convex_hull_roll_sum_2s'] = csv_df['dam_convex_hull'].rolling(roll_windows[4],
                                                                                      min_periods=1).sum()

            csv_df['head_convex_hull_roll_sum_1s'] = csv_df['head_convex_hull'].rolling(roll_windows[0],
                                                                                        min_periods=1).sum()
            csv_df['head_convex_hull_roll_sum_1ds'] = csv_df['head_convex_hull'].rolling(roll_windows[2],
                                                                                         min_periods=1).sum()
            csv_df['head_convex_hull_roll_sum_2s'] = csv_df['head_convex_hull'].rolling(roll_windows[4],
                                                                                        min_periods=1).sum()

            csv_df['high_p_pup_bp_roll_sum_1s'] = csv_df['high_p_pup_bp'].rolling(roll_windows[0], min_periods=1).sum()
            csv_df['high_p_pup_bp_roll_sum_1ds'] = csv_df['high_p_pup_bp'].rolling(roll_windows[2], min_periods=1).sum()
            csv_df['high_p_pup_bp_roll_sum_2s'] = csv_df['high_p_pup_bp'].rolling(roll_windows[4], min_periods=1).sum()

            csv_df['high_p_dam_bp_roll_sum_1s'] = csv_df['high_p_dam_bp'].rolling(roll_windows[0], min_periods=1).sum()
            csv_df['high_p_dam_bp_roll_sum_1ds'] = csv_df['high_p_dam_bp'].rolling(roll_windows[2], min_periods=1).sum()
            csv_df['high_p_dam_bp_roll_sum_2s'] = csv_df['high_p_dam_bp'].rolling(roll_windows[4], min_periods=1).sum()

            csv_df['nose_back10_length_roll_sum_1s'] = csv_df['nose_back10_length'].rolling(roll_windows[0],
                                                                                            min_periods=1).sum()
            csv_df['nose_back10_length_roll_sum_1ds'] = csv_df['nose_back10_length'].rolling(roll_windows[2],
                                                                                             min_periods=1).sum()
            csv_df['nose_back10_length_roll_sum_2s'] = csv_df['nose_back10_length'].rolling(roll_windows[4],
                                                                                            min_periods=1).sum()

            # Do unit correction
            print('Correcting coordinate units with', pixels_per_mm, 'pixels per mm')
            correction_bps = ['arm', 'side', 'dam_centroid', 'head_centroid', 'pups_centroid']
            correction_columns = [bp + '_x' for bp in correction_bps] + [bp + '_y' for bp in correction_bps]

            for col in correction_columns:
                csv_df[col] = csv_df[col] / pixels_per_mm

            # Save DF
            print('Saving features for video {}...'.format(file_name))
            self.data_df = csv_df
            self.data_df.columns = csv_df.columns
            self.data_df = self.data_df.fillna(0).apply(pd.to_numeric)
            save_path = os.path.join(self.save_dir, file_name + '.' + self.file_type)
            self.data_df = self.data_df.reset_index(drop=True).fillna(0)
            save_df(self.data_df, self.file_type, save_path)
            video_timer.stop_timer()
            print('Feature extraction complete for video {} (elapsed time: {}s)'.format(file_name, video_timer.elapsed_time_str))

        self.timer.stop_timer()
        stdout_success(f'Feature extraction complete for {str(len(self.files_found))} video(s). Results are saved inside the project_folder/csv/features_extracted directory', elapsed_time=self.timer.elapsed_time_str)
