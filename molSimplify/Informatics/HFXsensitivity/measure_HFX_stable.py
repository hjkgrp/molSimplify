import os
import glob
import numpy as np
import sys
import pandas as pd
import argparse
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, LeavePOut
from scipy import stats

'''
This script takes in an absolute path to a CSV file that has
complexes labeled, as well as exchange fractions. It then
takes those values and determines if the behavior is linear or
not. If so, it calculates the sensitivity. If not, then it
gives a reason for not computing it and logs that reason.

The script relies on raw data with one column labeled "complex_no_HFX"
and another labeled "alpha". The former contains the name with the ligand
field. The latter contains the HFX value.

If given no arguments, the function will just measure the sensitivity
of the spin splitting energies with an LOOCV cutoff of 5 kcal/mol,
requiring at least 4 points, and prioritizing lines with R2 of 0.99.
'''

def measure_sensitivity(path_to_csv, path_to_write=False, prop='SSE', R2_cutoff=0.99, CV_tolerance=5, num_points=4):
    if path_to_csv[0] != '/':
        path_to_csv = os.getcwd()+'/'+path_to_csv
    if path_to_write is False:
        path_to_write = os.getcwd()+'/'+os.path.split(path_to_csv)[1].replace('.csv','')
    raw_data = pd.read_csv(path_to_csv)
    raw_data = raw_data.sort_values(by=['complex_no_HFX','alpha'])

    ### This loops over unique ligand fields. Here, we keep track of things
    ### by compiling two lists. One is data that is kept and turned into a
    ### sensitivity. The other is any point that is eliminated. We log eliminations
    ### into two categories. The first is 'whole', which means that the whole
    ### ligand field is eliminated. The second is 'point', which means a single
    ### point was removed from the data point before measuring sensitivity.

    kept_dict_list = []
    removed_dict_list = []
    flag = False
    for i, unique_complex in raw_data.groupby('complex_no_HFX'):
        name_vals = unique_complex['complex'].tolist()
        alpha_vals = np.array(unique_complex['alpha'].tolist()).reshape(-1, 1)
        property_vals = np.array(unique_complex[str(prop)].tolist()).reshape(-1, 1)
        R2 = None
        #### First, we check if there are enough points. If not, we discard.
        if int(unique_complex.shape[0]) < num_points:
            for alpha, prop_val in zip(alpha_vals,property_vals):
                removed_dict_list.append({'complex_no_HFX':i, 'alpha':alpha[0], str(prop):prop_val[0],'reason':'not_enough_points_to_start','elim_type':'whole','R2':R2})
            continue

        ##### Next, we fit a line through the data points and check its R2 #####
        R2, reg = measure_R2(alpha_vals,property_vals)

        ##### If the R2 value is above the cutoff, we keep the data and do not process further #####
        if R2 >= R2_cutoff:
            for alpha, prop_val in zip(alpha_vals,property_vals):
                temp_dict = {'complex_no_HFX':i, 'alpha':alpha[0], str(prop):prop_val[0],'R2':R2,'sensitivity':float(reg.coef_)}
                kept_dict_list.append(temp_dict)
            continue
        else:
            ##### Next, we check for any points lying off of the line that can fix the line by removal of that point.
            ##### This check checks to see whether the removal of a single point results in the R2 test
            ##### being passed, or whether that point exceeds a heuristic cutoff.
            kept_points_X, kept_points_y, CV_removed_list = CV_check(alpha_vals, property_vals, name=i, prop=prop, CV_tolerance=CV_tolerance, R2_cutoff=R2_cutoff, num_points=num_points)
            new_R2, new_reg = measure_R2(kept_points_X,kept_points_y)
            if new_R2 >= R2_cutoff:
                # If removal of the point leads to the R2 test being passed, we stop processing.
                for alpha, prop_val in zip(kept_points_X,kept_points_y):
                    kept_dict_list.append({'complex_no_HFX':i, 'alpha':alpha[0], str(prop):prop_val[0],'R2':new_R2,'sensitivity':float(new_reg.coef_)})
                removed_dict_list += CV_removed_list
                continue
            else:
                previous = len(property_vals)
                while len(kept_points_X) >= num_points:
                    now = len(kept_points_X)
                    if now == previous:
                        break
                    kept_points_X, kept_points_y, new_removed_list = CV_check(kept_points_X, kept_points_y, name=i, prop=prop, CV_tolerance=CV_tolerance, R2_cutoff=R2_cutoff, num_points=num_points)
                    previous = len(kept_points_X)
                    CV_removed_list += new_removed_list

                ##### Next, we make sure the removal of the point allows us to have enough points. If not, we discard.
                if len(kept_points_X) < num_points:
                    for alpha, prop_val in zip(alpha_vals,property_vals):
                        removed_dict_list.append({'complex_no_HFX':i, 'alpha':alpha[0], str(prop):prop_val[0],'reason':'CV_resulted_in_not_enough_points','elim_type':'whole','R2':R2})
                    continue
                else:
                    ##### Next, we check the R2 again to see if the new points result in a better R2.
                    R2_repeat, reg_repeat = measure_R2(alpha_vals,property_vals)
                    if R2_repeat >= R2_cutoff:
                        for alpha, prop_val in zip(kept_points_X,kept_points_y):
                            print(alpha,prop_val)
                            kept_dict_list.append({'complex_no_HFX':i, 'alpha':alpha[0], str(prop):prop_val[0],'R2':R2_repeat,'sensitivity':float(reg_repeat.coef_)})
                        removed_dict_list += CV_removed_list
                        continue
                    else:
                        ##### If it does not meet the R2 check, we check the sign of the slopes.
                        kept_points_X, kept_points_y, slope_removed = slope_sign_check(kept_points_X,kept_points_y, name=i, prop=prop, num_points=num_points)
                        if len(kept_points_X) < num_points:
                            final_R2, reg_final = measure_R2(alpha_vals,property_vals)
                            for alpha, prop_val in zip(alpha_vals,property_vals):
                                removed_dict_list.append({'complex_no_HFX':i, 'alpha':alpha[0], str(prop):prop_val[0],'reason':'failed_sign_change_slope_check','elim_type':'whole','R2':final_R2})
                            continue
                        else:
                            # If we have enough points, we check the R2, and then repeat the outlier check if the line can be saved.
                            kept_R2, kept_reg = measure_R2(kept_points_X,kept_points_y)
                            if kept_R2 >= R2_cutoff:
                                for alpha, prop_val in zip(kept_points_X,kept_points_y):
                                    kept_dict_list.append({'complex_no_HFX':i, 'alpha':alpha[0], str(prop):prop_val[0],'R2':kept_R2,'sensitivity':float(kept_reg.coef_)})
                                removed_dict_list += slope_removed
                                continue
                            else:
                                kept_points_X, kept_points_y, CV_removed_list = CV_check(kept_points_X, kept_points_y, name=i, prop=prop, CV_tolerance=CV_tolerance, R2_cutoff=R2_cutoff, num_points=num_points)
                                backup_X = kept_points_X[:]
                                backup_y = kept_points_y[:]
                                previous = 10000
                                while len(kept_points_X) >= num_points:
                                    now = len(kept_points_X)
                                    if now == previous:
                                        break
                                    kept_points_X, kept_points_y, new_removed_list = CV_check(kept_points_X, kept_points_y, name=i, prop=prop, CV_tolerance=CV_tolerance, R2_cutoff=R2_cutoff, num_points=num_points)
                                    previous = len(kept_points_X)
                                    CV_removed_list += new_removed_list
                                if len(kept_points_X) < num_points:
                                    kept_points_X = backup_X
                                    kept_points_y = backup_y
                                else:
                                    removed_dict_list += CV_removed_list
                                R2, reg = measure_R2(kept_points_X,kept_points_y)
                                for alpha, prop_val in zip(kept_points_X,kept_points_y):
                                    temp_dict = {'complex_no_HFX':i, 'alpha':alpha[0], str(prop):prop_val[0],'R2':R2,'sensitivity':float(reg.coef_)}
                                    kept_dict_list.append(temp_dict)
                                continue
    #### Now we write all of our processed data to a dataframe.
    kept_data = pd.DataFrame(kept_dict_list)
    kept_data = kept_data[['complex_no_HFX','alpha',str(prop),'R2','sensitivity']]
    kept_data['symmetry'] = kept_data['complex_no_HFX'].apply(symmetry_class)
    kept_data = kept_data.sort_values(by=['R2','complex_no_HFX','alpha'])
    group_dict_list = []
    for i, group in kept_data.groupby('complex_no_HFX'):
        namelist = i.split('_')
        if namelist[2] != namelist[4]:
            symmetry = 'cis'
        elif (namelist[2]==namelist[4]) and (namelist[2]==namelist[6]) and (namelist[2]==namelist[7]):
            symmetry = 'homoleptic'
        elif (namelist[2]==namelist[4]) and (namelist[2]==namelist[6]) and (namelist[2]!=namelist[7]):
            symmetry = '5+1'
        elif (namelist[2]==namelist[4]) and (namelist[2]!=namelist[6]) and (namelist[2]!=namelist[7]) and (namelist[6]==namelist[7]):
            symmetry = 'trans'
        alphas = group['alpha'].tolist()
        energies = group[str(prop)].tolist()
        group_dict = {}
        group_dict['complex'] = i
        for j, val in enumerate([0, 5, 10, 15, 20, 25, 30]):
            if val not in alphas:
                group_dict[val] = np.nan
            else:
                idx = alphas.index(val)
                group_dict[val] = energies[idx]
        group_dict['symmetry'] = symmetry
        group_dict['sensitivity'] = group['sensitivity'].values[0]*100
        group_dict['R2'] = group['R2'].values[0]
        group_dict_list.append(group_dict)
    grouped_df = pd.DataFrame(group_dict_list)
    grouped_df = grouped_df[['complex',0, 5, 10, 15, 20, 25, 30,'R2','sensitivity','symmetry']]
    grouped_df.to_csv(path_to_write+'_kept_grouped.csv',index=False)
    thrown_data = pd.DataFrame(removed_dict_list)
    thrown_data = thrown_data[['complex_no_HFX','alpha',str(prop),'reason','elim_type','R2']]
    thrown_data['symmetry'] = thrown_data['complex_no_HFX'].apply(symmetry_class)
    thrown_data = thrown_data.sort_values(by=['complex_no_HFX','alpha'])
    group_dict_list = []
    for i, group in thrown_data.groupby('complex_no_HFX'):
        namelist = i.split('_')
        if namelist[2] != namelist[4]:
            symmetry = 'cis'
        elif (namelist[2]==namelist[4]) and (namelist[2]==namelist[6]) and (namelist[2]==namelist[7]):
            symmetry = 'homoleptic'
        elif (namelist[2]==namelist[4]) and (namelist[2]==namelist[6]) and (namelist[2]!=namelist[7]):
            symmetry = '5+1'
        elif (namelist[2]==namelist[4]) and (namelist[2]!=namelist[6]) and (namelist[2]!=namelist[7]) and (namelist[6]==namelist[7]):
            symmetry = 'trans'
        alphas = group['alpha'].tolist()
        energies = group[str(prop)].tolist()
        reasons = group['reason'].tolist()
        group_dict = {}
        group_dict['complex'] = i
        for j, val in enumerate([0, 5, 10, 15, 20, 25, 30]):
            if val not in alphas:
                group_dict[val] = np.nan
                group_dict[str(val)+'_elim'] = np.nan
            else:
                idx = alphas.index(val)
                group_dict[val] = energies[idx]
                group_dict[str(val)+'_elim'] = reasons[idx]
        group_dict['symmetry'] = symmetry
        group_dict['R2'] = group['R2'].values[0]
        group_dict_list.append(group_dict)
    grouped_df = pd.DataFrame(group_dict_list)
    grouped_df = grouped_df[['complex',0, 5, 10, 15, 20, 25, 30, '0_elim','5_elim','10_elim','15_elim','20_elim','25_elim','30_elim','symmetry','R2']]
    grouped_df.to_csv(path_to_write+'_elim_grouped.csv',index=False)
    kept_data['combined'] = kept_data['complex_no_HFX']+'_'+kept_data['alpha'].astype(str)
    thrown_data['combined'] = thrown_data['complex_no_HFX']+'_'+thrown_data['alpha'].astype(str)
    #### No points that are thrown away should also be kept.
    print('sanity check',set(kept_data['combined']).intersection(set(thrown_data['combined'])))
    #### Report how many ligand fields there were to start with.
    print(str(len(set(raw_data['complex_no_HFX'])))+' POSSIBLE sensitivities.')
    #### Check how many ligand fields end up being kept.
    print(str(len(set(kept_data['complex_no_HFX'])))+' FINAL calculated sensitivities.')
    #### Check how many whole lines are thrown out.
    whole = thrown_data[thrown_data['elim_type']=='whole']
    point = thrown_data[thrown_data['elim_type']=='point']
    print('ELIMINATED '+str(len(set(whole['complex_no_HFX'])))+' WHOLE ligand fields.')
    print('SAVED '+str(len(set(point['complex_no_HFX'])))+' ligand fields by eliminating a point or two.')
    #### Check how many ligand fields have something thrown out.
    print(str(len(set(thrown_data['complex_no_HFX'])))+' ligand fields with something removed.')

    #### Write all the data to CSVs.
    kept_data.to_csv(path_to_write+'_kept.csv')
    thrown_data.to_csv(path_to_write+'_discarded.csv')

def measure_R2(X, y):
    reg = LinearRegression()
    reg.fit(X, y)
    R2 = reg.score(X, y)
    return R2, reg

def symmetry_class(complex_name):
    namelist = complex_name.split('_')
    symmetry = 'unknown'
    if namelist[2] != namelist[4]:
        symmetry = 'cis'
    elif (namelist[2]==namelist[4]) and (namelist[2]==namelist[6]) and (namelist[2]==namelist[7]):
        symmetry = 'homoleptic'
    elif (namelist[2]==namelist[4]) and (namelist[2]==namelist[6]) and (namelist[2]!=namelist[7]):
        symmetry = '5+1'
    elif (namelist[2]==namelist[4]) and (namelist[2]!=namelist[6]) and (namelist[2]!=namelist[7]) and (namelist[6]==namelist[7]):
        symmetry = 'trans'
    return symmetry


def CV_check(X, y, name, prop, CV_tolerance, R2_cutoff, num_points):
    loo = LeaveOneOut()
    kept_points_X = False
    kept_points_y = False
    removed_dict_list = []
    originalR2, reg = measure_R2(X, y)
    ##### Perform LOOCV on the data with cutoffs provided #####
    for train_index, test_index in loo.split(X):
        train_X, test_X = X[train_index], X[test_index]
        train_y, test_y = y[train_index], y[test_index]
        ##### Fit the training data with a model and check its R2 #####
        R2, reg = measure_R2(train_X, train_y)
        if (R2 >= R2_cutoff) and len(train_X) >= num_points:# or (R2>originalR2):
            ##### If eliminating the single point improves the R2, keep that change.
            kept_points_X, kept_points_y = train_X, train_y
            removed_dict_list.append({'complex_no_HFX':name,'alpha':int(np.squeeze(test_X)), str(prop):float(np.squeeze(test_y)),'reason':'eliminating_point_led_to_R2_pass','elim_type':'point','R2':R2})
            return kept_points_X, kept_points_y, removed_dict_list
        else:
            pred_error = test_y - reg.predict(test_X)
            if (abs(pred_error)>CV_tolerance):
                kept_points_X, kept_points_y = train_X, train_y
                removed_dict_list.append({'complex_no_HFX':name,'alpha':int(np.squeeze(test_X)), str(prop):float(np.squeeze(test_y)),'reason':'point_had_LOOCV_greater_than_cutoff','elim_type':'point','R2':R2})
                return kept_points_X, kept_points_y, removed_dict_list
            if isinstance(kept_points_X, bool) or (len(kept_points_X)<num_points):
                kept_points_X, kept_points_y = X, y
    return kept_points_X, kept_points_y, removed_dict_list

def slope_sign_check(X, y, name, prop, num_points):
    kept_points_X = []
    kept_points_y = []
    elim_points_X = []
    elim_points_y = []
    num_slopes = len(X)-1
    coef_list = []
    removed_dict_list = []
    for i in range(num_slopes):
        reg = LinearRegression()
        temp_X = X[i:i+2]
        temp_y = y[i:i+2]
        reg.fit(temp_X,temp_y)
        coef_list.append(float(np.squeeze(reg.coef_)))
    neg_count = len(list(filter(lambda x: (x < 0), coef_list)))
    pos_count = len(list(filter(lambda x: (x >= 0), coef_list)))
    signchange = ((np.roll(np.sign(coef_list), 1) - np.sign(coef_list)) != 0).astype(int)
    signchange[0] = 0
    signchange_list = np.where(signchange==1)[0]/float(len(signchange))
    split_sign = np.array_split(signchange, 2)
    num_changes_first = np.sum(split_sign[0])
    num_changes_second = np.sum(split_sign[1])
    if len(signchange_list) == 0:
        sign_flag = 0
    else:
        sign_flag = signchange_list[0]
    diff_points = abs(neg_count-pos_count)
    remove_counter = 0
    if ((neg_count == pos_count) or (diff_points>=1 and len(X)<num_points) or
            ((len(X)-num_points-min(neg_count,pos_count)-1)<0 and (not min(neg_count,pos_count)<=1)) or ((sign_flag>0.4) and (sign_flag<0.6)) or (num_changes_first>0 and num_changes_second>0)):
        for j, val in enumerate(elim_points_X):
            removed_dict_list.append({'complex_no_HFX':name,'alpha':int(np.squeeze(val)), str(prop):float(np.squeeze(elim_points_y[j])),'reason':'identified_slope_sign_change','elim_type':'point'})
        return kept_points_X, kept_points_y, removed_dict_list
    else:
        for i in range(len(coef_list)-1):
            frac = float(i) / len(coef_list)
            if np.sign(coef_list[i]) != np.sign(coef_list[i+1]):
                if frac < 0.5:
                    kept_points_X = X[i+1:]
                    kept_points_y = y[i+1:]
                    elim_points_X = X[0:i+1]
                    elim_points_y = y[0:i+1]
                elif frac >= 0.5:
                    kept_points_X = X[0:i+1]
                    kept_points_y = y[0:i+1]
                    elim_points_X = X[i+1:]
                    elim_points_y = y[i+1:]
        if len(elim_points_X)>0:
            for j, val in enumerate(elim_points_X):
                removed_dict_list.append({'complex_no_HFX':name,'alpha':int(np.squeeze(val)), str(prop):float(np.squeeze(elim_points_y[j])),'reason':'identified_slope_sign_change_that_can_be_fixed','elim_type':'point', 'R2':np.nan})
    if (len(elim_points_X) == 0) or isinstance(elim_points_X,bool):
        kept_points_X = X
        kept_points_y = y
    return kept_points_X, kept_points_y, removed_dict_list


def main():
    parser = argparse.ArgumentParser(description='Script to process some sensitivity data.')
    parser.add_argument('--data', dest='path_to_csv', action='store', type=str, required=True,
                        help='Path to CSV containing raw data.')
    parser.add_argument('--writepath', dest='path_to_write', action='store', type=str,default=False,
                        help='Path to dump processed data. Defaults to dumping in script directory.')
    parser.add_argument('--prop', dest='prop', action='store', type=str, default='SSE',
                        help='Name for the property as in the CSV. Should be "SSE" for spin splitting.')
    parser.add_argument('--R2', dest='R2_cutoff', action='store', type=float, default=0.99,
                        help='R2 check cutoff value for linearity. Default is 0.99.')
    parser.add_argument('--cutoff', dest='CV_tolerance', action='store', type=int, default=5,
                        help='Heuristic cutoff for eliminating outliers. Defaults to 5 for SSE.')
    parser.add_argument('--num_points', dest='num_points', action='store', type=int, default=4,
                        help='Minimum number of points to form the HFX line. Defaults to 4.')
    args = parser.parse_args()
    print(args)
    measure_sensitivity(args.path_to_csv, args.path_to_write, args.prop, args.R2_cutoff, args.CV_tolerance, args.num_points)


if __name__ == "__main__":
    main()
