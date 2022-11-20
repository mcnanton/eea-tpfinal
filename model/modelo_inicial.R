# Setup----
library(dplyr)
library(tidymodels)
library(rsample)
library(bonsai)
library(DALEXtra)
tidymodels_prefer()

set.seed(1234)


# Levantamos particiones
dataset <- read.delim("datasets/WT_red_as_kmer.txt",
                      sep = ",") %>% 
  mutate(AS = as.factor(AS))

# train_data_total = read.csv("datasets/training_data.csv") %>% 
#   mutate(AS = as.factor(AS))
# train_data = read.csv("datasets/train_data.csv")  %>% 
#   mutate(AS = as.factor(AS))
# val_data = read.csv("datasets/val_data.csv")  %>% 
#   mutate(AS = as.factor(AS))
# test_data = read.csv("datasets/test_data.csv")  %>% 
#   mutate(AS = as.factor(AS))

dataset_split <- initial_split(dataset, prop = 0.80, strata = AS)
train_data_total <- training(dataset_split) #Partición 80%, no extraemos validación porque vamos a hacer cross-validation


# Biblioteca para lidiar con desbalanceos https://www.tidyverse.org/blog/2020/02/themis-0-1-0/


train_data_formula <-
  recipe(AS ~ ., data = train_data_total) #%>%
  #step_mutate(AS = factor(AS)) #esto puede reemplazar a los mutate de cada read.csv

#lgbm via {bonsai}
# https://github.com/tidymodels/bonsai
# https://bonsai.tidymodels.org/articles/bonsai.html
lgbm_a_tunear <- 
  boost_tree(
    trees = 200, 
    learn_rate = 0.05,
    tree_depth = tune(), 
    min_n = tune(),
    loss_reduction = 0, 
    mtry = tune()
    ) %>%
  set_engine(engine = "lightgbm") %>%
  set_mode(mode = "classification") %>% 
  translate()

# Esta función extrae los hiperparámetros a tunear 
lgbm_param_set <- extract_parameter_set_dials(lgbm_a_tunear) 

lgbm_wflow <- 
  workflow() %>% 
  add_model(lgbm_a_tunear) %>% 
  add_recipe(train_data_formula)

lgbm_wflow_tuning <- 
  lgbm_wflow %>% 
  extract_parameter_set_dials() %>% 
  finalize(train_data_total)

lgbm_wflow_tuning %>% extract_parameter_dials("mtry")

# CV
folds <- vfold_cv(train_data_total, v = 10)

# Param search ----
# https://www.tidymodels.org/learn/work/bayes-opt/

lgbm_bo <- 
  lgbm_wflow %>% 
  tune_bayes(
    resamples = folds, # Se tunea con validación cruzada
    # To use non-default parameter ranges
    param_info = lgbm_wflow_tuning,
    # Generate five at semi-random to start
    initial = 5,
    iter = 10,
    # How to measure performance?
    metrics = metric_set(roc_auc),
    control = control_bayes(no_improve = 30, verbose = TRUE)
  )

# Modelo con mejores parámetros de BO
show_best(lgbm_bo)
# mtry min_n tree_depth .metric .estimator  mean     n std_err .config              .iter
# <int> <int>      <int> <chr>   <chr>      <dbl> <int>   <dbl> <chr>                <int>
#   1  3911    25          5 recall  binary         1    10       0 Preprocessor1_Model1     0
# 2  1757     6         14 recall  binary         1    10       0 Preprocessor1_Model3     0
# 3   374    32          9 recall  binary         1    10       0 Preprocessor1_Model4     0
# 4  3035    19         10 recall  binary         1    10       0 Preprocessor1_Model5     0
# 5  4019    29          7 recall  binary         1    10       0 Iter2                    2
autoplot(lgbm_bo, type = "performance")

best_params <-
  lgbm_bo %>% 
  select_best(metric = "recall")

lgbm_final <- 
  boost_tree(
    #best_params #Esto no funcionó, por lo que los ingreso a mano
    trees = 200, 
    learn_rate = 0.05,
    tree_depth = best_params$tree_depth, 
    min_n = best_params$min_n,
    loss_reduction = 0, 
    mtry = best_params$mtry
  ) %>%
  set_engine(engine = "lightgbm") %>%
  set_mode(mode = "classification") %>% 
  translate()

# Predicción en test----
# Fiteamos el modelo en train_data_total y generamos predicción en test_data

full_dataset <- 
  recipe(AS ~ ., 
         data = dataset)

testing <- last_fit(lgbm_final, full_dataset, split = dataset_split)
testing %>% 
  collect_metrics()
# Importancia de variables enlatada

# SHAP values en test