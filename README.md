# Actionable path planning
The aim of this study is to plan an actionable path to change predicted values of a specified ML model.  
The license information and tutorials are described below.

# Tutorials
## Setting conda environment
Requirements:  
CentOS Linux release 8.1 (Confirmed)  
```bash
conda config --add channels conda-forge
conda create --name actionable_path_planning python=3.7.3
conda activate actionable_path_planning
conda install pandas=0.25.3=py37hb3f55d8_0
conda install numpy=1.16.0=py37h99e49ec_1
conda install scipy=1.3.2=py37h921218d_0
conda install xgboost=0.82=py37he1b5a44_1
conda install scikit-learn=0.21.2=py37hcdab131_1
conda install matplotlib=3.1.1=py37_0
conda install pystan=2.19.1.1=py37hb3f55d8_1
```

## For testing: Synthetic dataset generation
Run the following commands:
```bash
python synthetic_data_generation.py --dataset 5d --save_dir example_5d
```
- dataset: type of synthetic dataset. choices=[5d, 3d]  
- save_dir: save directory name

## Fitting ML model
```bash
python preprocessing.py --load_dir example_5d --model_name XG --model_type regressor
python fitting_ml_model.py --load_dir example_5d
```
- load_dir: load directory name, which should contains `df_X.csv`, `df_y.csv`, `df_var.csv` in `data` subdirectory  
-- df_var.csv: including `item_name_other` and `item_type` information for each variable. `item_type` choices=[continuous, nominal]  
- model_name: ML algorithm. choices=[XG: XGBoost, RF: RandomForest, SVM: SupportVectorMachine]  
- model_type: choices=[regressor, classifier]

## Stochastic surrogate modeling
```bash
python surrogate_modeling.py --load_dir example_5d --sigma_y 1.27 --stan_template template_for_wbic.stan
python surrogate_modeling_result_postprocessing.py --load_dir example_5d
```
- load_dir: load directory name  
- sigma_y: modeling hyperparameter. Recommended setting is rmse/2.  
- stan_template: `template_for_wbic_classifier.stan` should be selected for classification task  

The number of mixture components in hierarchical Bayesian model with the lowest WBIC would be output.

## Path planning
```bash
python selection_for_path_planning.py --load_dir example_5d --mixture_components 2
python path_planning.py --load_dir example_5d --mixture_components 2 --num_movable 5
python calc_actionability_score.py --load_dir example_5d --mixture_components 2 --num_movable 5
```
- load_dir: load directory name
- path_planning_index: indication of instance index to plan path. In default setting, all instances w/o outlier.  
- intervention_variables: indication of explanatory variables selected as intervention variables. In default setting, all explanatory variables.  
- mixture_components: the number of mixture components of hierarchical Bayesian model to be used for path planning.  
- destination_state: search end condition. choices=[count, criteria(Not supported)]  
- destination: if `destination_state` is count, search iteration count, elif criteria, y value to be achieved.  
- step: unit change in intervention variables. Default setting is 0.5-sigma of training dataset.  
- upper_is_better: increased y state would be better in search setting or not.  
- num_movable: number of intervention variables to distinguish loading directory.  

The output would contains the planned path for each instance and the actionability score.  

# License
A patent is pending.  
This edition of Actionable path planning is for evaluation, learning, and non-profit academic research purposes only, and a license is needed for any other uses. Please send requests on license or questions to nakamura.kazuki.88m[at]st.kyoto-u.ac.jp
