#!/usr/bin/env python
# coding: utf-8

import numpy  as np
import pandas as pd
import re

# 離散変数
def insert_disc_to_stan_file(template_stan_text, 
                             x_disc
                             ):
    
    output_stan_text = template_stan_text
    
    output_stan_text = _insert_disc_text_to_data_section(
        output_stan_text, x_disc)

    output_stan_text = _insert_disc_text_to_parameters_section(
        output_stan_text, x_disc)

    output_stan_text = _insert_disc_text_to_model_section_prior(
        output_stan_text, x_disc)

    output_stan_text = _insert_disc_text_to_model_section_likelihood(
        output_stan_text, x_disc)

    output_stan_text = _insert_disc_text_to_generated_quantities_section(
        output_stan_text, x_disc)

    return output_stan_text


def _insert_disc_text_to_data_section(template_stan_text, x_disc):
    output_stan_text = template_stan_text
    list_text = []
    
    for var_index in x_disc.index:
        list_text.append('    int<lower=1> X_disc_{}[N];'.format(var_index))
        list_text.append('    int<lower=1> n_cat_{};'.format(var_index))
        list_text.append('    vector<lower=0>[n_cat_{}] alpha_disc_{};'.format(var_index, var_index))
    
    additional_text = "\n".join(list_text)
    
    output_stan_text = re.sub(r'// discrete data start //\s*// discrete data end //',
                              additional_text, output_stan_text)
    
    return output_stan_text


def _insert_disc_text_to_parameters_section(template_stan_text, x_disc):
    output_stan_text = template_stan_text
    list_text = [] 
    
    for var_index in x_disc.index:
        list_text.append('    simplex[n_cat_{}] phi_X_disc_{}[K];'.format(var_index, var_index))
    
    additional_text = "\n".join(list_text)
    
    output_stan_text = re.sub(r'// discrete parameters start //\s*// discrete parameters end //',
                              additional_text, output_stan_text)
    
    return output_stan_text


def _insert_disc_text_to_model_section_prior(template_stan_text, x_disc):
    output_stan_text = template_stan_text
    list_text = []
    
    for var_index in x_disc.index:
        list_text.append('        phi_X_disc_{}[k] ~ dirichlet(alpha_disc_{});'.format(var_index, var_index))
    
    additional_text = "\n".join(list_text)
    
    output_stan_text = re.sub(r'// discrete model prior start //\s*// discrete model prior end //',
                              additional_text, output_stan_text)
    
    return output_stan_text


def _insert_disc_text_to_model_section_likelihood(template_stan_text, x_disc):
    output_stan_text = template_stan_text
    list_text = []
    
    for var_index in x_disc.index:
        list_text.append('            eta[k] += categorical_lpmf(X_disc_{}[n] | phi_X_disc_{}[k]);'.format(var_index, var_index))
    
    additional_text = "\n".join(list_text)
    
    output_stan_text = re.sub(r'// discrete model likelihood start //\s*// discrete model likelihood end //',
                              additional_text, output_stan_text)
    
    return output_stan_text


def _insert_disc_text_to_generated_quantities_section(template_stan_text, x_disc):
    output_stan_text = template_stan_text
    list_text = [] 
    
    for var_index in x_disc.index:
        list_text.append('            eta += categorical_lpmf(X_disc_{}[n] | phi_X_disc_{}[k]);'.format(var_index, var_index))
    
    additional_text = "\n".join(list_text)
    
    output_stan_text = re.sub(r'// discrete generated quantities start //\s*// discrete generated quantities end //',
                              additional_text, output_stan_text)
    
    return output_stan_text


def insert_zero_poi_to_stan_file(template_stan_text, 
                                 x_zero_poi 
                                 ):
    
    output_stan_text = template_stan_text
    
    output_stan_text = _insert_zero_poi_text_to_data_section(
        output_stan_text, x_zero_poi)

    output_stan_text = _insert_zero_poi_text_to_parameters_section(
        output_stan_text, x_zero_poi)

    output_stan_text = _insert_zero_poi_text_to_model_section_prior(
        output_stan_text, x_zero_poi)

    output_stan_text = _insert_zero_poi_text_to_model_section_likelihood(
        output_stan_text, x_zero_poi)

    output_stan_text = _insert_zero_poi_text_to_generated_quantities_section(
        output_stan_text, x_zero_poi)

    return output_stan_text


def _insert_zero_poi_text_to_data_section(template_stan_text, x_zero_poi):
    output_stan_text = template_stan_text
    list_text = []
    
    for var_index in x_zero_poi.index:
        list_text.append('    int<lower=0> X_zero_poi_{}[N];'.format(var_index))

    additional_text = "\n".join(list_text)
    
    output_stan_text = re.sub(r'// zero poi data start //\s*// zero poi data end //',
                              additional_text, output_stan_text)
    
    return output_stan_text


def _insert_zero_poi_text_to_parameters_section(template_stan_text, x_zero_poi):
    output_stan_text = template_stan_text
    list_text = []
    
    for var_index in x_zero_poi.index:
        list_text.append('    real<lower=0> lambda_X_zero_poi_{}[K];'.format(var_index))
        list_text.append('    simplex[2] phi_X_zero_poi_{}[K];'.format(var_index))

    additional_text = "\n".join(list_text)
    
    output_stan_text = re.sub(r'// zero poi parameters start //\s*// zero poi parameters end //',
                              additional_text, output_stan_text)
    
    return output_stan_text


def _insert_zero_poi_text_to_model_section_prior(template_stan_text, x_zero_poi):
    output_stan_text = template_stan_text
    list_text = []
    
    for var_index in x_zero_poi.index:
        list_text.append('        phi_X_zero_poi_{}[k] ~ dirichlet(alpha_bern);'.format(var_index))
        list_text.append('        lambda_X_zero_poi_{}[k] ~ cauchy(0, 5);'.format(var_index))

    additional_text = "\n".join(list_text)
    
    output_stan_text = re.sub(r'// zero poi model prior start //\s*// zero poi model prior end //',
                              additional_text, output_stan_text)
    
    return output_stan_text


def _insert_zero_poi_text_to_model_section_likelihood(template_stan_text, x_zero_poi):
    output_stan_text = template_stan_text
    list_text = []
    
    for var_index in x_zero_poi.index:
        list_text.append('            if (X_zero_poi_{}[n]==0)'.format(var_index) + '{')
        list_text.append('                eta[k] += log_sum_exp(')
        list_text.append('                    bernoulli_lpmf(0 | phi_X_zero_poi_{}[k][2]),'.format(var_index))
        list_text.append('                    bernoulli_lpmf(1 | phi_X_zero_poi_{}[k][2]) + poisson_lpmf(0 | lambda_X_zero_poi_{})'.format(var_index, var_index))
        list_text.append('                );')
        list_text.append('            } else {')
        list_text.append('                eta[k] += bernoulli_lpmf(1 | phi_X_zero_poi_{}[k][2]) + poisson_lpmf(X_zero_poi_{}[n] | lambda_X_zero_poi_{}[k]);'.format(var_index, var_index, var_index))
        list_text.append('            }')

    additional_text = "\n".join(list_text)
    
    output_stan_text = re.sub(r'// zero poi model likelihood start //\s*// zero poi model likelihood end //',
                              additional_text, output_stan_text)
    
    return output_stan_text


def _insert_zero_poi_text_to_generated_quantities_section(template_stan_text, x_zero_poi):
    output_stan_text = template_stan_text
    list_text = []
    
    for var_index in x_zero_poi.index:
        list_text.append('            if (X_zero_poi_{}[n]==0)'.format(var_index) + '{')
        list_text.append('                eta += log_sum_exp(')
        list_text.append('                    bernoulli_lpmf(0 | phi_X_zero_poi_{}[k][2]),'.format(var_index))
        list_text.append('                    bernoulli_lpmf(1 | phi_X_zero_poi_{}[k][2]) + poisson_lpmf(0 | lambda_X_zero_poi_{})'.format(var_index, var_index))
        list_text.append('                );')
        list_text.append('            } else {')
        list_text.append('                eta += bernoulli_lpmf(1 | phi_X_zero_poi_{}[k][2]) + poisson_lpmf(X_zero_poi_{}[n] | lambda_X_zero_poi_{}[k]);'.format(var_index, var_index, var_index))
        list_text.append('            }')

    additional_text = "\n".join(list_text)
    
    output_stan_text = re.sub(r'// zero poi generated quantities start //\s*// zero poi generated quantities end //',
                              additional_text, output_stan_text)
    
    return output_stan_text