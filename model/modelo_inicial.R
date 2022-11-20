# Setup
library(dplyr)
library(tidymodels)
library(rsample)
library(bonsai)
tidymodels_prefer()

# Levantamos particiones
train_data_total = read.csv("datasets/training_data.csv") %>% 
  mutate(AS = as.factor(AS))
train_data = read.csv("datasets/train_data.csv")  %>% 
  mutate(AS = as.factor(AS))
val_data = read.csv("datasets/val_data.csv")  %>% 
  mutate(AS = as.factor(AS))
test_data = read.csv("datasets/test_data.csv")  %>% 
  mutate(AS = as.factor(AS))

# Biblioteca para lidiar con desbalanceos https://www.tidyverse.org/blog/2020/02/themis-0-1-0/


preprocesamiento <-
  recipe(AS ~ ., data = train_data_total) #%>%
  #step_mutate(AS = factor(AS))

#lgbm via {bonsai}
# https://github.com/tidymodels/bonsai
dt_mod <- 
  boost_tree(
    trees = 100, 
    learn_rate = 0.05,
    tree_depth = tune(), 
    min_n = tune(),
    loss_reduction = tune(), 
    mtry = tune()
  ) %>%
  set_engine(engine = "lightgbm") %>%
  set_mode(mode = "classification") %>% 
  translate()

# tidy_workflow <- 
#   workflow() %>%
#   add_recipe(preprocesamiento) %>% 
#   add_model(dt_mod) #%>% 
#   #extract_parameter_set_dials() %>% 
#   #finalize(train_data_total)

# Esta función extrae los hiperparámetros a tunear 
lgbm_param_set <- extract_parameter_set_dials(dt_mod) 

lgbm_wflow <- 
  workflow() %>% 
  add_model(dt_mod) %>% 
  add_recipe(preprocesamiento)

lgbm_param <- 
  lgbm_wflow %>% 
  extract_parameter_set_dials() %>% 
  finalize(train_data_total)

lgbm_param %>% extract_parameter_dials("mtry")

# CV
folds <- vfold_cv(train_data_total, v = 10)

# Param search 
# https://www.tidymodels.org/learn/work/bayes-opt/

lgbm_bo <- 
  lgbm_wflow %>% 
  tune_bayes(
    resamples = folds,
    # To use non-default parameter ranges
    param_info = lgbm_param,
    # Generate five at semi-random to start
    initial = 5,
    iter = 25,
    # How to measure performance?
    metrics = metric_set(roc_auc),
    control = control_bayes(no_improve = 30, verbose = TRUE)
  )
