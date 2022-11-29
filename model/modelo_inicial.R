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
  mutate(AS = as.factor(AS, levels=c('1','0')))

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


# Tenemos un desbalanceo fuerte
# Biblioteca para lidiar con desbalanceos https://www.tidyverse.org/blog/2020/02/themis-0-1-0/
# Un ejemplo del workflow con step_downsample https://www.r-bloggers.com/2020/05/bayesian-hyperparameters-optimization/
# https://recipes.tidymodels.org/reference/step_downsample.html


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

lgbm_wflow_tuning %>% extract_parameter_dials("mtry") # mtry depende directamente del dataset

# CV
folds <- vfold_cv(train_data_total, v = 10, strata = AS)

# Param search ----
# https://www.tidymodels.org/learn/work/bayes-opt/

lgbm_bo <- 
  lgbm_wflow %>% 
  tune_bayes(
    resamples = folds, # Se tunea con validación cruzada
    # To use non-default parameter ranges
    param_info = lgbm_wflow_tuning,
    # Generate five at semi-random to start
    initial = 5, #It is suggested that the number of initial results be greater than the number of parameters being optimized
    iter = 20,
    # How to measure performance?
    metrics = metric_set(roc_auc),
    control = control_bayes(no_improve = 30, verbose = TRUE)
  )

# Modelo con mejores parámetros de BO
show_best(lgbm_bo)
# mtry min_n tree_depth .metric .estimator  mean     n std_err .config .iter
# <int> <int>      <int> <chr>   <chr>      <dbl> <int>   <dbl> <chr>   <int>
#   1   720    39         10 roc_auc binary     0.690    10  0.0270 Iter9       9
# 2   962    38         13 roc_auc binary     0.686    10  0.0282 Iter3       3
# 3   648    35          9 roc_auc binary     0.686    10  0.0268 Iter11     11
# 4    44    14          8 roc_auc binary     0.679    10  0.0280 Iter17     17
# 5    93    35          6 roc_auc binary     0.679    10  0.0386 Iter10     10

#Si la métrica a optimizar es recall:
# mtry min_n tree_depth .metric .estimator  mean     n std_err .config              .iter
# <int> <int>      <int> <chr>   <chr>      <dbl> <int>   <dbl> <chr>                <int>
#   1  3454    40          7 recall  binary         1    10       0 Preprocessor1_Model2     0
# 2  2631    12         14 recall  binary         1    10       0 Preprocessor1_Model3     0
# 3   639    28          5 recall  binary         1    10       0 Preprocessor1_Model4     0
# 4  1916    24         12 recall  binary         1    10       0 Preprocessor1_Model5     0
# 5   330    39          1 recall  binary         1    10       0 Iter1                    1

autoplot(lgbm_bo, type = "performance")

best_params <-
  lgbm_bo %>% 
  select_best(metric = "recall")

best_params
# mtry min_n tree_depth .config
# <int> <int>      <int> <chr>  
#   1   720    39         10 Iter9 

# Si la metrica es recall
# mtry min_n tree_depth .config             
# <int> <int>      <int> <chr>               
#   1  3454    40          7 Preprocessor1_Model2

lgbm_final <- 
  boost_tree(
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

testing <- last_fit(lgbm_final, full_dataset, split = dataset_split) #DANIEL avisame si ves ok esto. Por lo que veo aca entiendo que sí https://tune.tidymodels.org/reference/last_fit.html

testing %>% 
  collect_metrics()

#Estas metricas son para una BO optimizada por roc_auc, pero son identicas para recall
# .metric  .estimator .estimate .config             
# <chr>    <chr>          <dbl> <chr>               
#   1 accuracy binary         0.996 Preprocessor1_Model1
# 2 roc_auc  binary         0.655 Preprocessor1_Model1

predicciones <- 
  testing %>% 
  collect_predictions() 
# El modelo no esta teniendo sensibilidad a la clase 1, no predice ningun AS = 1

cm <- testing %>%
  conf_mat(obs, pred)

autoplot(cm, type = "heatmap")

# Importancia de variables enlatada

# SHAP values en test
# https://community.rstudio.com/t/shap-values-with-tidymodels/147217
# https://www.tmwr.org/explain.html