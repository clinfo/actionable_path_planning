import os
import time
from functools import wraps
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

def make_dir(dir_name):
    if os.path.exists(dir_name)==False:
        os.mkdir(dir_name)

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        print(f"[INFO] start {func.__name__}")
        start = time.time()
        result = func(*args, **kargs)
        elapsed_time = time.time() - start
        print(f"[INFO] done in {elapsed_time:5f} s")
        return result
    return wrapper

def softmax(a):
    """ softmax
    Args:
        a (np.array[float]): 
    Return:
        y (np.array[float]):
    """      
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def my_round(val, digit=0):
    """ round
    pythonの組み込みのroundは偶数への丸めを行ってしまう．
    Args:
        val (float): value
        digit (int): 小数点以下何点を保持するか
    Return: rounded value (float)
    """      
    p = 10 ** digit
    return (val * p * 2 + 1) // 2 / p

def show_steps(nodes_on_path, xy_for_stan, file_name, model_type='regressor'):
    """
    Args:
        nodes_on_path (list[Node]): 
        xy_for_stan (class): 
        file_name (str): 
    Return:
    """
    with open(file_name, 'w') as f:
        previous_r_x = None
        count = 0
        nodes_on_path.reverse()
        for node in nodes_on_path:
            current_r_x = np.array(node.r_x)
            if previous_r_x is not None:
                dif_r_x = current_r_x - previous_r_x
                if sum(dif_r_x)==1:
                    stepped_feature = xy_for_stan.x_reg.name[np.where(dif_r_x==1)[0][0]]
                    if model_type=='regressor':
                        f.write(f'[step {count}, y={my_round(node.y,2)}] +1 {stepped_feature}\n')
                    elif model_type=='classifier':
                        f.write(f'[step {count}, y={my_round(node.y_pred_prob,2)}] +1 {stepped_feature}\n')
                    else:
                        raise NotImplementedError()
                elif sum(dif_r_x)==-1:
                    stepped_feature = xy_for_stan.x_reg.name[np.where(dif_r_x==-1)[0][0]]
                    if model_type=='regressor':
                        f.write(f'[step {count}, y={my_round(node.y,2)}] -1 {stepped_feature}\n')
                    elif model_type=='classifier': 
                        f.write(f'[step {count}, y={my_round(node.y_pred_prob,2)}] -1 {stepped_feature}\n')
                    else:
                        raise NotImplementedError()
            else:
                if model_type=='regressor':
                    f.write(f'[step {count}, y={my_round(node.y,2)}]\n')
                elif model_type=='classifier':
                    f.write(f'[step {count}, y={my_round(node.y_pred_prob,2)}]\n')
                else:
                    raise NotImplementedError()
            previous_r_x = current_r_x
            count += 1
        f.close()
    return

def x_coords_summary(initial_node, destination_node, xy_for_stan, file_name):
    '''
    Args:
        initial_node (Node):
        destination_node (Node):
        xy_for_stan (class):
        file_name (str):
    Returns:
    '''
    cont_mean = xy_for_stan.x_cont.standard.mean
    cont_std = xy_for_stan.x_cont.standard.std
    num_x_cont = len(cont_mean)
    
    with open(file_name, 'a') as f:
        f.write('\n')
        f.write('-----------------------------------------------------------\n')
        f.write(f'x_features: {xy_for_stan.x_reg.name}\n')
        f.write(f'destination_r_x_coords: {destination_node.r_x}\n')
        f.write(f'initial_x_coords: {initial_node.x.reg.array}\n')
        f.write(f'destination_x_coords: {destination_node.x.reg.array}\n')
        f.write('\n')
        initial_x_cont_raw = (initial_node.x.reg.array[:num_x_cont] * cont_std) + cont_mean
        f.write(f'initial_x_coords (cont, raw): {initial_x_cont_raw}')
        destination_x_cont_raw = (destination_node.x.reg.array[:num_x_cont] * cont_std) + cont_mean
        f.write(f'destination_x_coords (cont, raw): {destination_x_cont_raw}')
        f.write('-----------------------------------------------------------\n')
        f.close()
    return

def calc_dif_distance(df_result):
    '''
    Attribute:
        Actionability (dif_distance) score calculation
    Args:
        df_result (pd.DataFrame): 
    Returns:
        dif_distance (pd.Series):
    '''
    min_distance = df_result['min_distance']
    rand_distance = df_result.drop('min_distance', axis=1)
    rand_distance_mean = rand_distance.mean(axis=1)
    dif_distance = rand_distance_mean - min_distance
    return dif_distance

def plot_dif_distance(dif_distance, save_dir, n_bins=25, outlier_value=0, fontsize=24):
    '''
    Plot actionability score
    Args:
        dif_distance (pd.Series):
        save_dir (str):
        n_bins (int): num of bins
        outlier_value (int):
    Returns:
    '''
    if outlier_value!=0:
        dif_distance.loc[dif_distance>=outlier_value] = outlier_value    
    
    plt.figure(figsize=(8,8))
    n, bins, _ = plt.hist(dif_distance, bins=n_bins, align='mid', ec='black')
    idx = np.argmax(n)
    plt.axvline(dif_distance.median(), color='r', linestyle='--')
    plt.legend({'Median = {:.2f}'.format(dif_distance.median()): dif_distance.median()}, fontsize=fontsize*0.75)
    plt.yticks(fontsize=fontsize*0.75)
    plt.xticks(fontsize=fontsize*0.75)
    plt.ylim([-0.001*n.max(), n.max()*1.05])
    plt.xlabel('Actionability Score', fontsize=fontsize)
    plt.ylabel('Number of Instances', fontsize=fontsize)

    for axis in ['top', 'bottom', 'left', 'right']:
        plt.gca().spines[axis].set_linewidth(4)

    plt.savefig(save_dir + 'actionability_score.png')
    plt.close()
    return

