# Setup----
library(dplyr)
library(tidymodels)
library(rsample)
library(bonsai)
library(DALEXtra)
library(themis)
tidymodels_prefer()

set.seed(1234)

########### OBJETIVO: PREDECIR VARIABLE AS ########### 

# Levantamos particiones
dataset <- read.delim("datasets/WT_red_as_kmer_all.txt",
                      sep = ",") %>% 
  mutate(AS = factor(AS, levels=c('1','0')))

dataset_split <- initial_split(dataset, prop = 0.80, strata = AS)
train_data_total <- training(dataset_split) #Partición 80%, no extraemos validación porque vamos a hacer cross-validation


# Tenemos un desbalanceo fuerte
# Biblioteca para lidiar con desbalanceos https://www.tidyverse.org/blog/2020/02/themis-0-1-0/
# Un ejemplo del workflow con step_downsample https://www.r-bloggers.com/2020/05/bayesian-hyperparameters-optimization/
# https://recipes.tidymodels.org/reference/step_downsample.html


train_data_formula <-
  recipe(AS ~ ., data = train_data_total) %>%
  step_downsample(AS, under_ratio = 5)
#step_mutate(AS = factor(AS)) #esto puede reemplazar a los mutate de cada read.csv

#lgbm via {bonsai}
# https://github.com/tidymodels/bonsai
# https://bonsai.tidymodels.org/articles/bonsai.html
lgbm_a_tunear <- 
  boost_tree(
    trees = 300, 
    learn_rate = tune(),#0.05,
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
folds <- vfold_cv(train_data_total, v = 5, strata = AS)
# Ejemplo de undersampling con CV https://www.tidymodels.org/learn/models/sub-sampling/
# En este ejemplo el df es el dataset de entrenamiento
# NO ESTOY SEGURA DE QUE ESTO ESTÉ FUNCIONANDO BIEN (que la CV sea sobre el dataset undersampleado)

# Param search ----
# https://www.tidymodels.org/learn/work/bayes-opt/

ctrl <- control_bayes(verbose = TRUE)

lgbm_bo <- 
  lgbm_wflow %>% 
  tune_bayes(
    resamples = folds, # Se tunea con validación cruzada # NO ESTOY SEGURA DE QUE ESTO ESTÉ FUNCIONANDO BIEN (que la CV sea sobre el dataset undersampleado)!!!
    # To use non-default parameter ranges
    param_info = lgbm_wflow_tuning,
    # Generate five at semi-random to start
    initial = 5, #It is suggested that the number of initial results be greater than the number of parameters being optimized
    iter = 15,
    # How to measure performance?
    metrics = metric_set(pr_auc),
    control = ctrl
  )

# Modelo con mejores parámetros de BO
show_best(lgbm_bo)

#Mejores hiperparámetros optimizando por PR_AUC
# mtry min_n tree_depth   learn_rate .metric .estimator  mean     n std_err .config .iter
# <int> <int>      <int>        <dbl> <chr>   <chr>      <dbl> <int>   <dbl> <chr>   <int>
#   1  4046    21          1 0.000000295  pr_auc  binary     0.248     5  0.0197 Iter8       8
# 2  4035    26          1 0.000000210  pr_auc  binary     0.248     5  0.0197 Iter13     13
# 3  4053    20          1 0.0000000626 pr_auc  binary     0.248     5  0.0197 Iter14     14
# 4  4066    23          1 0.00000294   pr_auc  binary     0.241     5  0.0212 Iter15     15
# 5  3920    22          1 0.00000251   pr_auc  binary     0.230     5  0.0130 Iter6       6


autoplot(lgbm_bo, type = "performance")

best_params <-
  lgbm_bo %>% 
  select_best(metric = "pr_auc")

best_params
# mtry min_n tree_depth  learn_rate .config
# <int> <int>      <int>       <dbl> <chr>  
#   1  4046    21          1 0.000000295 Iter8  

lgbm_final <- 
  boost_tree(
    trees = 300, 
    learn_rate = 0.000000295, #best_params$learn_rate,
    tree_depth = 1, #best_params$tree_depth, 
    min_n = 21, #best_params$min_n,
    loss_reduction = 0, 
    mtry = 4046 #best_params$mtry
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

#Estas metricas son para una BO optimizada por PR_AUC
# .metric  .estimator .estimate .config             
# <chr>    <chr>          <dbl> <chr>               
#   1 accuracy binary         0.969 Preprocessor1_Model1
# 2 roc_auc  binary         0.642 Preprocessor1_Model1

predicciones <- 
  testing %>% 
  collect_predictions() 
# El modelo no esta teniendo sensibilidad a la clase 1, no predice ningun AS = 1

# cm <- testing %>% #Esto no funciona
#  conf_mat(obs, pred)

autoplot(cm, type = "heatmap")

df_predicciones <- as.data.frame(testing$.predictions)

violin_plot = ggplot(df_predicciones, aes(x=AS, y=.pred_1, group=AS, fill=factor(AS))) + 
  geom_violin() +
  theme_bw() +
  guides(scale="none", fill=guide_legend(title="AS"))

violin_plot



# Importancia de variables enlatada

# SHAP values en test
# https://community.rstudio.com/t/shap-values-with-tidymodels/147217
# https://www.tmwr.org/explain.html