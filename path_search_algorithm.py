import copy
import itertools
from math import inf as infinity
import numpy as np
from scipy import stats, special

from utils import timeit, softmax
import stan_base_class as stanb

class XCoords(object):
    """ XCoords(super class)
    Attributes:
        reg (ArrayName): regression
        cont (XFeatures): continuous features under Gaussian distribution
        disc (XFeaturesDisc): discrete features
        zerp_poi (XFeatures): continuous features under zero_poi distribution
    """
    def __init__(self):
        """ A constructor of XCoords
        Args:
        """
        self.reg = stanb.ArrayName()
        self.cont = stanb.XFeatures()
        self.disc = stanb.XFeaturesDisc()
        self.zero_poi = stanb.XFeatures()

class InitialXCoords(XCoords):
    """ InitialXCoords
    Attributes:
        reg (ArrayName): initial x_coords of cypher for regression.
        cont (XFeatures): initial x_coords of cypher for continuous.
        disc (XFeaturesDisc): initial x_coords of cypher for discrete.
        zerp_poi (XFeatures): initial x_coords of cypher for zero_poi.
    """
    def __init__(self, xy_for_stan, initial_idx):
        """ A constructor of InitialXCoords
        Args:
            xy_for_stan (class of stan_dataset): dataset for stan
            initial_idx (int): cypher_i in xy_for_stan to search
        """        
        super().__init__()
        self.num_cdz = [len(xy_for_stan.x_cont.clus.name),
                        len(xy_for_stan.x_disc.clus.name),
                        len(xy_for_stan.x_zero_poi.clus.name)]
        self.reg.array = xy_for_stan.x_reg.array[initial_idx]
        self.reg.name = xy_for_stan.x_reg.name
        self.cont.clus.array = xy_for_stan.x_cont.clus.array[initial_idx]
        self.cont.clus.name = xy_for_stan.x_cont.clus.name
        if xy_for_stan.x_disc.clus.array is not None:
            self.disc.reg.name = xy_for_stan.x_disc.reg.name
            self.disc.clus.name = xy_for_stan.x_disc.clus.name
            self.disc.clus.n_cat = xy_for_stan.x_disc.clus.n_cat
            self.disc.clus.array = xy_for_stan.x_disc.clus.array[initial_idx]
            self.disc.index = xy_for_stan.x_disc.index
        if xy_for_stan.x_zero_poi.clus.array is not None:
            self.zero_poi.clus.name = xy_for_stan.x_zero_poi.clus.name
            self.zero_poi.clus.array = xy_for_stan.x_zero_poi.clus.array[initial_idx]
            self.zero_poi.index = xy_for_stan.x_zero_poi.index
        
class StepXCoords(XCoords):
    """ StepXCoords
    Attributes:
        reg (ArrayName): stepped x_coords of cypher for regression.
        cont (XFeatures): stepped x_coords of cypher for continuous.
        disc (XFeaturesDisc): stepped x_coords of cypher for discrete.
        zerp_poi (XFeatures): stepped x_coords of cypher for zero_poi.
    """
    def __init__(self, initial_x_coords, r_x, step=0.1):
        """ A constructor of StepXCoords
        Args:
            initial_x_coords (InitialXCoords): initial x_coords of cypher.
            r_x (tuple[int]): relative x_coords to initial x_coords
            step (float): to what extent step x_coords by relative x_coords
        """    
        super().__init__()
        # deepcopy initial_x_coords
        self.num_cdz = initial_x_coords.num_cdz
        self.reg = copy.deepcopy(initial_x_coords.reg)
        self.cont = copy.deepcopy(initial_x_coords.cont)
        self.disc = copy.deepcopy(initial_x_coords.disc)
        self.zero_poi = copy.deepcopy(initial_x_coords.zero_poi)
        self.reg.array += np.array(r_x) * step
        self.cont.clus.array += np.array(r_x[sum(initial_x_coords.num_cdz[:0]):\
                                             sum(initial_x_coords.num_cdz[:1])]) * step
        if initial_x_coords.disc.clus.array is not None:
            self.disc.clus.array += np.array(r_x[sum(initial_x_coords.num_cdz[:1]):\
                                                 sum(initial_x_coords.num_cdz[:2])]) * 1
        if initial_x_coords.zero_poi.clus.array is not None:
            self.zero_poi.clus.array += np.array(r_x[sum(initial_x_coords.num_cdz[:2]):\
                                                     sum(initial_x_coords.num_cdz[:3])]) * step

class XFixed:
    """ Fixed
    Attributes:
        cont (np.array[boolean of 0/1]): 1d-array of fixed continuous features
        disc_reg (np.array[boolean of 0/1]): 1d-array of fixed discrete features
        disc (np.array[boolean of 0/1]): 1d-array of fixed discrete features
        zero_poi (np.array[boolean of 0/1]): 1d-array of fixed zero_poi features
    """
    def __init__(self):
        """ A constructor of XFixed
        Args:
        """    
        self.cont = np.array([])
        self.disc_reg = np.array([])
        self.disc = np.array([])
        self.zero_poi = np.array([])
        self.reg = np.array([])
    
    def set_fixed(self, initial_node, 
                  fixed_cont=None,
                  fixed_disc_reg=None,
                  fixed_disc=None,
                  fixed_zero_poi=None):
        """ set fixed feature
        If not directed (None)，define np.zeros (all features are not fixed).
        Args:
            fixed_cont (np.array[boolean or 0/1]): 1d-array of fixed continuous features
            fixed_disc_reg (np.array[boolean or 0/1]): 1d-array of fixed discrete features for regression
            fixed_disc (np.array[boolean or 0/1]): 1d-array of fixed discrete features
            fixed_zero_poi (np.array[boolean or 0/1]): 1d-array of fixed zero_poi features
        """    
        if fixed_cont is None:
            self.cont = np.zeros(initial_node.x.cont.clus.array.shape[0])
        else:
            self.cont = fixed_cont
        # update self.reg
        self.reg = np.hstack([self.reg, self.cont])
        if initial_node.x.disc.reg.name is not None:
            if fixed_disc_reg is None:
                self.disc_reg = np.zeros(len(initial_node.x.disc.reg.name))
            else:
                self.disc_reg = fixed_disc_reg
            # update self.reg
            self.reg = np.hstack([self.reg, self.disc_reg])
        if initial_node.x.disc.clus.array is not None:
            if fixed_disc is None:
                self.disc = np.zeros(initial_node.x.disc.clus.array.shape[0])
            else:
                self.disc = fixed_disc
        if initial_node.x.zero_poi.clus.array is not None:
            if fixed_zero_poi is None:
                self.zero_poi = np.zeros(initial_node.x.zero_poi.clus.array.shape[0])        
            else:
                self.zero_poi = fixed_zero_poi
            # update self.reg
            self.reg = np.hstack([self.reg, self.zero_poi])

class Node:
    """ Node
    Attributes: 
        r_x (tuple[int]): relative x_coords
        x (XCoords): class for x_coords
        x_fixed (XFixed): class for x_fixed
        y (float): y_coords
        class_lp (np.array[float]): 'log(p(X,Y|class) * p(class))' of xy_coords
        neg_logprob (float) : '-log(Σp(X,Y|class) * p(class))' of xy_coords  
        tentative_distance (float) : tentative cost to this node
        c_visited (boolean) : visited flag during calc tentative_distance
        f_visited (boolean) : visited flag during find path
          
    """
    def __init__(self, initial_x_coords, r_x, step=0.1):
        """ A constructor of Node
        Args: 
            initial_x_coords (InitialXCoords): initial x_coords of cypher
            r_x (tuple[int]): ralative x_coords．list or np.array cannot be added to set, but tuple can.
            step (float): to what extent step x_coords by relative x_coords
        """
        self.r_x = r_x
        self.x = StepXCoords(initial_x_coords, self.r_x, step)
        self.x_fixed = XFixed()
        self.y = None
        self.class_lp = None
        self.neg_logprob = infinity
        self.tentative_distance = infinity
        self.c_visited = False
        self.f_visited = False
    
    def set_y_and_class_lp(self, model, y_type, k_class, xy_for_stan, dict_emp_bayes, sigma_y, p_class, model_type='regressor'):
        """ Calculate and set y from x_coords of current_node
        Args:
            model (model): model to explain
            y_type (str): {'pred', 'glmm'}
            k_class (int): num of latent class
            xy_for_stan (class of stan_dataset): dataset for stan
            dict_emp_bayes (dict): empirical bayes parameters
            sigma_y (int/float): σ for y regression
            p_class (np.array[float]): p(class)
            model_type (str): choices = ['regressor', 'classifier']
        Return:
        """
        if y_type=='pred':
            # to predict y, reshape to 2d-array is needed
            # [0]: to convert np.array() to values
            self.y = model.predict(self.x.reg.array.reshape(1,self.x.reg.array.shape[0]))[0]                
            self.class_lp = self.calc_class_lp(self.y, k_class, xy_for_stan, dict_emp_bayes, sigma_y, p_class, model_type)
        elif y_type=='glmm': # [Not bug fixed]
            y_pred = model.predict(self.x.reg.array.reshape(1,self.x.reg.array.shape[0]))[0]
            self.class_lp = self.calc_class_lp(y_pred, k_class, xy_for_stan, dict_emp_bayes, sigma_y, p_class, model_type)
            self.y = self.calc_y_glmm(dict_emp_bayes)
        if model_type=='classifier':
            self.y_pred_prob = model.predict_proba(self.x.reg.array.reshape(1,self.x.reg.array.shape[0]))[0, 1] 
            
            
    def calc_class_lp(self, y, k_class, xy_for_stan, dict_emp_bayes, sigma_y, p_class, model_type='regressor', epsilon=1e-6):
        """ Calculate class_lp of node
        calculate class_lp (=log(p(X,Y | class) * p(class))) from X, Y
        Args:
            y (float) : predicted y
            k_class (int): num of latent class．
            xy_for_stan (class of stan_dataset): dataset for stan
            dict_emp_bayes (dict): empirical bayes parameters
            sigma_y (int or float): σ for y regression
            p_class (np.array[float]): p(class)
            model_type (str): choices = ['regressor', classifier]
            epsilon (float): to deal with extreme cov.
        Return:
            class_lp (np.array[float]): log(p(X,Y | class) * p(class))
        """
        # for stats calculation
        e = np.eye(k_class)
        dict_e_disc_idx = {}
        for disc_idx, n_cat in zip(xy_for_stan.x_disc.index, xy_for_stan.x_disc.clus.n_cat):
            dict_e_disc_idx[disc_idx] = np.eye(n_cat)
        # calculate log prob
        class_lp = stats.multinomial.logpmf(e, n=1, p=p_class)
        mu = dict_emp_bayes['beta1'] + np.dot(dict_emp_bayes['beta2'], self.x.reg.array) # k*v, v
        if model_type=='regressor': # regressor
            class_lp += stats.norm.logpdf(y, loc=mu, scale=sigma_y)
        elif model_type=='classifier': # classifier
            class_lp += stats.bernoulli.logpmf(y, p=special.expit(mu))
        else:
            print('[ERROR] Unimplemented task type.')
        for i in range(len(self.x.cont.clus.array)):
            class_lp += stats.norm.logpdf(self.x.cont.clus.array[i],
                                          loc=dict_emp_bayes['phi_mu_X_cont'][:,i],
                                          scale=np.sqrt(dict_emp_bayes['diag_sigma_X_cont'][:,i]+epsilon))
        # discrete
        for i, disc_idx in enumerate(xy_for_stan.x_disc.index):
            x_disc = int(self.x.disc.clus.array[i]-1)
            class_lp += stats.multinomial.logpmf(dict_e_disc_idx[disc_idx][x_disc],
                                            n=1, p=dict_emp_bayes[f'phi_X_disc_{disc_idx}'])
        # zero_poi
        for i, zero_poi_idx in enumerate(xy_for_stan.x_zero_poi.index):
            x_zero_poi = self.x.zero_poi.clus.array[i]
            prob = dict_emp_bayes[f'phi_X_zero_poi_{zero_poi_idx}'][:,1]
            lamb = dict_emp_bayes[f'lambda_X_zero_poi_{zero_poi_idx}']
            if x_zero_poi==0:
                class_lp += special.logsumexp([stats.bernoulli.logpmf(0, p=prob),
                                               stats.bernoulli.logpmf(1, p=prob) + stats.poisson.logpmf(0, lamb)])
            else:
                class_lp += stats.bernoulli.logpmf(1, p=prob) + stats.poisson.logpmf(x_zero_poi, lamb)   
        
        return class_lp
    
    def calc_y_glmm(self, dict_emp_bayes):
        """ Calculate y_glmm of node
        calculate
        gamma = p(class | X,Y) = softmax(class_lp)
        from
        class_lp = log(p(X,Y | class) * p(class)) 
        and calculate y_glmm by gamma.
        Args:
            dict_emp_bayes (dict): empirical bayes parameters.
        Return:
            y_glmm (float): y_glmm calculated by gamma and empirical bayes parameters.
        """
        gamma = softmax(self.class_lp)
        # mixed intercept
        beta1_mixed = np.dot(gamma, dict_emp_bayes['beta1'])
        # mixed coef
        beta2_mixed = np.dot(gamma, dict_emp_bayes['beta2'])
        # mixed regression
        y_glmm = beta1_mixed + np.dot(self.x.reg.array, beta2_mixed)
        return y_glmm
    
    def set_neg_logprob(self):
        """ Calculate neg_logprob of node
        calculate and set neg_log_prob from xy_coords of Node.
        neg_log_prob = -log(Σp(X,Y | class) * p(class))
        Args:
        Return:
        """
        self.neg_logprob = -special.logsumexp(self.class_lp)
    
    def distance_to(self, neighbor_node):
        """ Get cost to neighbor_node
        Get cost to neighbor_node from current_node.
        Cost is neg_log_prob of neighbor_node, regardless of current_node
        Args:
            neighbor_node (Node): neighbor_node of current_node
        Return: neg_logprob of neighbor_node
        """
        return neighbor_node.neg_logprob
    
    def exit_check(self, c, destination_state, destination, unvisited, upper_is_better, max_count, model_type='regressor'):
        """ Check if current_node satisfy the exit conditions
        Args:
            c (int): iteration count
            destination_state (str): type of destination state. {'criteria', 'count'}
            destination (int/float or tuple[int]): if destination state is...
                                                   ...criteria, target y_coords (int/float)
                                                   ...count, maximum counts (int)
            unvisited (set[node]): unvisited nodes
            upper_is_better (boolean): 
            max_count (int): 
            model_type (str): choices = ['regressor', 'classifier']
        Return: True if current_node satisfy the exit conditions or no nodes in unvisited, else False.
        """
        if (self not in unvisited):
            print(f'\n[INFO] All nodes were searched')
            return True
        elif c == max_count:
            print(f'\n[INFO] Did not reach criteria in {max_count} counts')
            return True        
        elif destination_state == 'criteria': 
            if model_type=='regressor':
                if upper_is_better:
                    if self.y >= destination:
                        print(f'\n[INFO] Reached destination')
                        return True
                else:
                    if self.y <= destination:
                        print(f'\n[INFO] Reached destination')
                        return True
            elif model_type=='classifier':
                if self.y_pred_prob <= destination:
                    print(f'\n[INFO] Reached destination')
                    return True
            else:
                print('[ERROR] Unimplemented task type.')
        elif destination_state == 'count':
            if c == destination:
                print(f'\n[INFO] Iteration count search has been done')
                return True
        else:
            raise NotImplementedError()
        return False
    
    def destination_check(self, destination_node, best_y, smallest_tentative_distance, upper_is_better, model_type='regressor'):
        """ Check if selected node is destination_node
        if current_node is more appropriate than tentative node, update destination_node.
        Args:
            destination_node (Node): tentative destination_node
            best_y (float): tentative best y_coords
            smallest_tentative_distance (float): smallest tentative_distance of best_y
            upper_is_better
            model_type (str): choices = ['regressor', 'classifier']
        Return: 
            destination_node (Node): updated destination_node
            best_y (float): updated best_y
            smallest_tentative_distance (float): updated smallest_tentative_distance
        """
        if self.c_visited:
            if model_type=='regressor':
                if upper_is_better:
                    if (self.y > best_y)&(self.neg_logprob < infinity):
                        destination_node = self
                        best_y = self.y
                        smallest_tentative_distance = self.tentative_distance
                    elif (self.y == best_y)&(self.tentative_distance < smallest_tentative_distance):
                        destination_node = self
                        smallest_tentative_distance = self.tentative_distance
                else:
                    if (self.y < best_y)&(self.neg_logprob < infinity):
                        destination_node = self
                        best_y = self.y
                        smallest_tentative_distance = self.tentative_distance
                    elif (self.y == best_y)&(self.tentative_distance < smallest_tentative_distance):
                        destination_node = self
                        smallest_tentative_distance = self.tentative_distance
            elif model_type=='classifier':
                if upper_is_better:
                    if (self.y_pred_prob > best_y)&(self.neg_logprob < infinity):
                        destination_node = self
                        best_y = self.y_pred_prob
                        smallest_tentative_distance = self.tentative_distance
                    elif (self.y_pred_prob == best_y)&(self.tentative_distance < smallest_tentative_distance):
                        destination_node = self
                        smallest_tentative_distance = self.tentative_distance
                else:
                    if (self.y_pred_prob < best_y)&(self.neg_logprob < infinity):
                        destination_node = self
                        best_y = self.y_pred_prob
                        smallest_tentative_distance = self.tentative_distance
                    elif (self.y_pred_prob == best_y)&(self.tentative_distance < smallest_tentative_distance):
                        destination_node = self
                        smallest_tentative_distance = self.tentative_distance
            else:
                print('[ERROR] Not implemented task type.')
        return destination_node, best_y, smallest_tentative_distance
    
    def calc_p_class(self, update_flag, dict_emp_bayes):
        """ calcurate p_class
        If update_flag is True, update p_class value according to fixed x values.
        p(class=k | X) = p(X | class=k) * p(class=k) / Σ(p(X | class=j) * p(class=j))
        Args:
            update_flag (boolean): If False, use original p_class
            dict_emp_bayes (dict): empirical bayes parameters
        Return: 
            p_class (np.array[float]): updated p_class
        """
        p_class = dict_emp_bayes['pi']
        if update_flag==False:
            return p_class
        else:
            class_lp = np.log(p_class)
            class_lp = _calc_p_class_cont(class_lp, self.x.cont, self.x_fixed.cont, dict_emp_bayes)
            class_lp = _calc_p_class_disc(class_lp, self.x.disc, self.x_fixed.disc, dict_emp_bayes)
            class_lp = _calc_p_class_zero_poi(class_lp, self.x.zero_poi, self.x_fixed.zero_poi, dict_emp_bayes)
            p_class = softmax(class_lp)
            p_class = p_class / (p_class.sum() + 1e-10)
        return p_class

    
def _calc_p_class_cont(class_lp, x_cont, fixed, dict_emp_bayes):
    """ sub function to calc update class_lp for continuous features
    Args:
        class_lp (np.array[float]): current class_lp
        x_cont (XFeatures): 
        fixed (array[boolean or 0/1]): 1d-array of fixed features
        dict_emp_bayes (dict): empirical bayes parameters
    Return: 
        class_lp (np.array[float]): updated class_lp
    """
    for i, f in enumerate(fixed):
        if f==1: # 1 or True is OK...
            class_lp += stats.norm.logpdf(x_cont.clus.array[i],
                                          loc=dict_emp_bayes['phi_mu_X_cont'][:,i], 
                                          scale=np.sqrt(dict_emp_bayes['diag_sigma_X_cont'][:,i]))
        else:
            continue
    return class_lp

def _calc_p_class_disc(class_lp, x_disc, fixed, dict_emp_bayes):
    """ sub function to calc update class_lp for discrete features
    Args:
        class_lp (np.array[float]): current class_lp
        x_disc (XFeaturesDisc): 
        fixed (array[boolean or 0/1]): 1d-array of fixed features
        dict_emp_bayes (dict): empirical bayes parameters
    Return: 
        class_lp (np.array[float]): updated class_lp
    """
    for i, f in enumerate(fixed):
        if f==1: 
            disc_index = x_disc.index[i]
            np_eye = np.eye(x_disc.clus.n_cat[i])
            X_disc = np_eye[int(x_disc.clus.array[i]-1)]
            class_lp += stats.multinomial.logpmf(X_disc,
                                                 n=1,
                                                 p=dict_emp_bayes[f'phi_X_disc_{disc_index}'])
        else:
            continue
    return class_lp

def _calc_p_class_zero_poi(class_lp, x_zero_poi, fixed, dict_emp_bayes):
    """ sub function to calc update class_lp for zero_poi features
    Args:
        class_lp (np.array[float]): current class_lp
        x_zero_poi (XFeatures): 
        fixed (array[boolean or 0/1]): 1d-array of fixed features
        dict_emp_bayes (dict): empirical bayes parameters
    Return: 
        class_lp (np.array[float]): updated class_lp
    """
    for i, f in enumerate(fixed):
        if f==1:
            zero_poi_index = x_zero_poi.index[i]
            prob = dict_emp_bayes[f'phi_X_zero_poi_{zero_poi_index}'][:,1]
            lamb = dict_emp_bayes[f'lambda_X_zero_poi_{zero_poi_index}']
            X_zero_poi = x_zero_poi.clus.array[i].astype(int)
            if X_zero_poi==0:
                class_lp += special.logsumexp([stats.bernoulli.logpmf(0, p=prob),
                                               stats.bernoulli.logpmf(1, p=prob) + stats.poisson.logpmf(0, lamb)])
            else:
                class_lp += stats.bernoulli.logpmf(1, p=prob) + stats.poisson.logpmf(X_zero_poi, lamb)    
        else:
            continue
    return class_lp


class Graph:
    """ Graph
    Attributes:
        initial_node (Node): initial_node
        nodes (list[Node]): nodes to search. Nodes are added during search.
        list_r_x (list[tuple[int]]): relative x_coords of search area, which is also used to get index of node.
        destination_state (str): type of destination state. {'node', 'criteria', 'count'}
        destination (int/float or tuple[int]): if destination state is...
                                               ...criteria, target y_coords (int/float)
                                               ...count, maximum counts (int)
        dict_emp_bayes (dict): empirical bayes parameters
        k_class (int) : num of latent class．
        xy_for_stan (class of stan_dataset): dataset for stan
        model (model): model to explain
        sigma_y (int or float): σ to y regression
        p_class (np.array[float]): p(class)
        step (float): to what extent step x_coords by relative x_coords
        max_count (int): 
        upper_is_better (boolean):
    """
    def __init__(self, initial_node, destination_state, destination, dict_emp_bayes, p_class, xy_for_stan, k_class, sigma_y, upper_is_better, step=0.1, max_count=1000, y_type='pred'):
        """ A constructor of Graph
        Args: 
            initial_node (Node): initial_node
            destination_state (str): type of destination state. {'node', 'criteria', 'count'}
            destination (int/float or tuple[int]): if destination state is...
                                                   ...criteria, target y_coords (int/float)
                                                   ...count, maximum counts (int)
            dict_emp_bayes (dict): empirical bayes parameters
            p_class (np.array[float]): p(class)
            xy_for_stan (class of stan_dataset): dataset for stan
            k_class (int) : num of latent class．
            sigma_y (int or float): σ for y regression
            upper_is_better (boolean):
            step (float): to what extent step x_coords by relative x_coords
            max_count (int): 
            y_type (str): {'pred', 'glmm'}
        """
        self.initial_node = initial_node
        self.nodes = [initial_node]
        self.list_r_x = [initial_node.r_x]
        self.destination_state = destination_state
        self.destination = destination
        self.dict_emp_bayes = dict_emp_bayes
        self.k_class = k_class
        self.xy_for_stan = xy_for_stan
        self.model = xy_for_stan.model
        self.sigma_y = sigma_y
        self.p_class = p_class
        self.step = step
        self.y_type = y_type
        self.upper_is_better = upper_is_better
        self.max_count = max_count

    def get_neighbors(self, node, unvisited, make_flag, model_type='regressor'):
        """ Get neighbor_nodes of current_node
        Get neighbor_node using relative x_coords.
        If make_flag is True，generate new node for neighbor_node.
        Generated node is added to nvisited, since tentative_distance is unknown.
        Args:
            node (Node): current_node
            unvisited (set[Node]): nodes which tentative_distance is not confirmed.
            make_flag (boolean): if True, if neighbor_node does not exist, make neighbor_node.
            model_type (str): choices = ['regressor', 'classifier']
        Return: 
            neightors (list[Node]): neighbor_node list of current_node
            unvisited (set[Node]): nodes which tentative_distance is not confirmed.
        """
        neighbors = []
        for x_d in itertools.product(range(len(self.initial_node.x.reg.name)), [-1,1]):
            if self.initial_node.x_fixed.reg[x_d[0]]==0: 
                n_r_x = copy.deepcopy(np.array(node.r_x))
                n_r_x[x_d[0]] += x_d[1] # index 0: x_i , 1: direction. 
                n_r_x = tuple(n_r_x)
                if n_r_x in self.list_r_x: 
                    n_node = self.nodes[self.list_r_x.index(n_r_x)]
                elif make_flag:
                    n_node = Node(self.initial_node.x, n_r_x, self.step)
                    n_node.set_y_and_class_lp(self.model, self.y_type, self.k_class, self.xy_for_stan,
                                              self.dict_emp_bayes, self.sigma_y, self.p_class, model_type)
                    n_node.set_neg_logprob()
                    if n_node.neg_logprob != infinity:
                        self.nodes.append(n_node)
                        self.list_r_x.append(n_node.r_x)
                        unvisited.add(n_node)
#                         print(f'[INFO] Relative coords of new node: {n_node.r_x}')
                if n_node.neg_logprob != infinity:
                    neighbors.append(n_node)
        return neighbors, unvisited

    @timeit        
    def breadth_first_calc_distance(self, model_type='regressor'):
        """ Calculate distance by breadth_first method
        Args: 
            model_type (str): choices=['regressor', 'classifier']
        Retern: 
            destination_node (Node): destination_node
        """
        initial_node = self.initial_node
        initial_node.set_y_and_class_lp(self.model, self.y_type, self.k_class, self.xy_for_stan,
                                        self.dict_emp_bayes, self.sigma_y, self.p_class, model_type)
        initial_node.tentative_distance = 0 # set 0 to tentative_distance of initial_node
        unvisited = set(self.nodes) # substantially, only initial_node
        current_node = initial_node
        done = False # finish flag
        c = 0 # count
        while not done:
            c += 1
#             print(f'\n[INFO] Count: {c}')
            # get neighbor_nodes of current_node.
            # if new x_coords, make new node and add it to nodes and unvisited. list_r_x is also updated.
            # if tentative_distance of neighbor_nodes via current_node is current one, update
            neighbors, unvisited = self.get_neighbors(current_node, unvisited, make_flag=True, model_type=model_type)
            for neighbor_node in neighbors:
                if not neighbor_node or neighbor_node.c_visited: # not neighbor means no neighbor_node.
                    continue
                new_tentative_distance = current_node.tentative_distance + current_node.distance_to(neighbor_node)
                if neighbor_node.tentative_distance > new_tentative_distance:
                    neighbor_node.tentative_distance = new_tentative_distance

            # Since tentative distance of current_node is minimal distance,
            # check if current_node satisfy the destination state．
            done = current_node.exit_check(c, self.destination_state, self.destination, unvisited,
                                           self.upper_is_better, self.max_count, model_type)
            if done:
                current_node.c_visited = True
                if c==self.max_count:
                    return None
                # if destination_state is 'node' or 'criteria', destination_node is current_node
                elif (self.destination_state=='node')|(self.destination_state=='criteria'):
                    destination_node = current_node
                # if destination_state is 'count' or no nodes in unvisited, search destination_node in nodes.
                elif (self.destination_state=='count')|(current_node not in unvisited):
                    smallest_tentative_distance = infinity
                    destination_node = self.initial_node
                    if model_type=='regressor':
                        best_y = self.initial_node.y
                    elif model_type=='classifier':
                        best_y = self.initial_node.y_pred_prob
                    else:
                        raise NotImplementedError()
                    for node in self.nodes:
                        # if satisfy the conditions, update
                        destination_node, best_y, smallest_tentative_distance = \
                            node.destination_check(destination_node, best_y, 
                                                   smallest_tentative_distance, self.upper_is_better, model_type)
                break
                
            # tentative_distance of current_node is now decided, c_visited is True and remove from unvisited
            # if a node is removed from unvisited, it never becomes a curren_node.
            current_node.c_visited = True
            unvisited.remove(current_node)            
            # Select next node. Next node is a node which have smallest tentative_distance in unvisited.
            smallest_tentative_distance = infinity
            for node in unvisited:
                # if unvisited is empty or tentative_distance of any nodes in unvisited is infinity,
                # current_node is not updated, and search will be finished at next loop,
                # since current_node is no more in unvisited.
                if node and node.tentative_distance < smallest_tentative_distance:
                    smallest_tentative_distance = node.tentative_distance
                    current_node = node
        
        return destination_node
    
    def breadth_first_find_path(self, destination_node):
        """ Find path by breadth_first method
        Find path from destination_node to initial_node.
        Args:
            destination_node (Node): destination_node
        Return:
            nodes_on_path (list[Node]): Nodes on best path.
        """
        initial_node = self.initial_node
        # Go from destination node to initial node to find path
        current_node = destination_node
        current_node.f_visited = True
        smallest_tentative_distance = destination_node.tentative_distance
        nodes_on_path = [current_node]
        # move on to neighbor_node which has minimal tentative_distance from current_node
        while current_node is not initial_node:
            neighbors, _ = self.get_neighbors(current_node, unvisited=set(), make_flag=False, model_type=None)
            for neighbor_node in neighbors:
                if not neighbor_node or neighbor_node.f_visited:
                    continue
                if neighbor_node.tentative_distance < smallest_tentative_distance:
                    smallest_tentative_distance = neighbor_node.tentative_distance
                    neighbor_node.f_visited = True
                    current_node = neighbor_node
            nodes_on_path.append(current_node)        
        return nodes_on_path
