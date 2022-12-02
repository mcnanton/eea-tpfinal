# Setup----
library(dplyr)
library(tidymodels)
library(rsample)
library(bonsai)
library(DALEXtra)
library(themis)

tidymodels_prefer()

set.seed(1234)

# Levantamos particiones
dataset <- read.delim("datasets/WT_red_as_kmer_all.txt",
                      sep = ",") %>% 
  mutate(AS = factor(AS, levels=c('1','0')))

dataset_split <- initial_split(dataset, prop = 0.80, strata = AS)
train_data_total <- training(dataset_split) #Partición 80%, no extraemos validación porque vamos a hacer cross-validation
test_data <- testing(dataset_split) #%>% dplyr::mutate_all(as.factor)

# Biblioteca para lidiar con desbalanceos https://www.tidyverse.org/blog/2020/02/themis-0-1-0/
# Un ejemplo del workflow con step_downsample https://www.r-bloggers.com/2020/05/bayesian-hyperparameters-optimization/
# https://recipes.tidymodels.org/reference/step_downsample.html


train_data_formula <-
  recipe(AS ~ ., data = train_data_total) %>%
  step_downsample(AS, 
                  under_ratio = 2,
                  seed = 42)

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


# Param search ----
# https://www.tidymodels.org/learn/work/bayes-opt/

ctrl <- control_bayes(verbose = TRUE)

lgbm_bo <- 
  lgbm_wflow %>% 
  tune_bayes(
    resamples = folds, # Se tunea con validación cruzada 
    # To use non-default parameter ranges
    param_info = lgbm_wflow_tuning,
    # Generate five at semi-random to start
    initial = 10, #It is suggested that the number of initial results be greater than the number of parameters being optimized
    iter = 20,
    # How to measure performance?
    metrics = metric_set(pr_auc),
    control = ctrl
  )

# Modelo con mejores parámetros de BO
show_best(lgbm_bo)

#Mejores hiperparámetros optimizando por RECALL, under_ratio = 1 seed 42 downsample
# mtry min_n tree_depth learn_rate .metric .estimator  mean     n std_err .config               .iter
# <int> <int>      <int>      <dbl> <chr>   <chr>      <dbl> <int>   <dbl> <chr>                 <int>
#   1    39    28         11    0.0904  recall  binary     0.732     5  0.0137 Iter6                     6
# 2   266    29          9    0.00437 recall  binary     0.717     5  0.0257 Preprocessor1_Model02     0
# 3   681     3          6    0.00377 recall  binary     0.716     5  0.0251 Iter11                   11
# 4   357     2         10    0.0555  recall  binary     0.713     5  0.0126 Iter3                     3
# 5  1134    20          7    0.0875  recall  binary     0.711     5  0.0208 Iter13                   13

#Mejores hiperparámetros optimizando por RECALL, under_ratio = 2 seed 42 downsample
# mtry min_n tree_depth learn_rate .metric .estimator  mean     n std_err .config .iter
# <int> <int>      <int>      <dbl> <chr>   <chr>      <dbl> <int>   <dbl> <chr>   <int>
#   1  3848    27          2     0.0892 recall  binary     0.359     5  0.0268 Iter16     16
# 2   548    17          3     0.0520 recall  binary     0.344     5  0.0223 Iter20     20
# 3  3865     5          4     0.0980 recall  binary     0.342     5  0.0251 Iter15     15
# 4  2636    38          5     0.0824 recall  binary     0.335     5  0.0164 Iter19     19

#Mejores hiperparámetros optimizando por RECALL, under_ratio = 15 seed 42 downsample
# mtry min_n tree_depth learn_rate .metric .estimator    mean     n std_err .config               .iter
# <int> <int>      <int>      <dbl> <chr>   <chr>        <dbl> <int>   <dbl> <chr>                 <int>
#   1  4093    13          5     0.0616 recall  binary     0.00697     5 0.00239 Iter18                   18
# 2  4079    26          8     0.0661 recall  binary     0.00566     5 0.00277 Iter8                     8
# 3  3935    22          6     0.0734 recall  binary     0.00547     5 0.00138 Iter16                   16
# 4  4053     3          3     0.0814 recall  binary     0.00542     5 0.00247 Iter6                     6
# 5  3862    24         14     0.0496 recall  binary     0.00415     5 0.00170 Preprocessor1_Model01     0


#Mejores hiperparámetros optimizando por PR_AUC, under_ratio = 2
# mtry min_n tree_depth  learn_rate .metric .estimator  mean     n std_err .config .iter
# <int> <int>      <int>       <dbl> <chr>   <chr>      <dbl> <int>   <dbl> <chr>   <int>
#   1  3874    36          1 0.00000242  pr_auc  binary     0.319     5  0.0185 Iter14     14
# 2  4030    40          1 0.000000847 pr_auc  binary     0.319     5  0.0185 Iter17     17
# 3  3997    39          1 0.000000166 pr_auc  binary     0.303     5  0.0255 Iter18     18
# 4  3812    38          1 0.00000874  pr_auc  binary     0.282     5  0.0216 Iter20     20
# 5  3604    27          1 0.00000208  pr_auc  binary     0.257     5  0.0133 Iter13     13

#Mejores hiperparámetros optimizando por PR_AUC, under_ratio = 15
# mtry min_n tree_depth   learn_rate .metric .estimator  mean     n std_err .config .iter
# <int> <int>      <int>        <dbl> <chr>   <chr>      <dbl> <int>   <dbl> <chr>   <int>
# mtry min_n tree_depth learn_rate .metric .estimator   mean     n std_err .config               .iter
# <int> <int>      <int>      <dbl> <chr>   <chr>       <dbl> <int>   <dbl> <chr>                 <int>
#   1   789    40         11    0.0160  pr_auc  binary     0.0793     5 0.00721 Iter7                     7
# 2   615    37         11    0.00970 pr_auc  binary     0.0789     5 0.00865 Iter15                   15
# 3   891    25         10    0.0119  pr_auc  binary     0.0761     5 0.00605 Iter17                   17
# 4   266    29          9    0.00437 pr_auc  binary     0.0757     5 0.00536 Preprocessor1_Model02     0
# 5   838    40         14    0.00799 pr_auc  binary     0.0745     5 0.00462 Iter8                     8


autoplot(lgbm_bo, type = "performance")

best_params <-
  lgbm_bo %>% 
  select_best(metric = "pr_auc")

best_params

#Optimizando por recall, undersampling 1
# mtry min_n tree_depth learn_rate .config
# <int> <int>      <int>      <dbl> <chr>  
#   1    39    28         11     0.0904 Iter6 

#Optimizando por recall, undersampling 2
# mtry min_n tree_depth learn_rate .config
# <int> <int>      <int>      <dbl> <chr>  
#   1  3848    27          2     0.0892 Iter16 

#Optimizando por recall, undersampling 15
# mtry min_n tree_depth learn_rate .config
# <int> <int>      <int>      <dbl> <chr>  
#   1  4093    13          5     0.0616 Iter18 

#Optimizando por pr_auc, undersampling 2
# mtry min_n tree_depth learn_rate .config
# <int> <int>      <int>      <dbl> <chr>  
#   1  3874    36          1 0.00000242 Iter14 

#Optimizando por pr_auc
# mtry min_n tree_depth learn_rate .config
# <int> <int>      <int>      <dbl> <chr>  
#   1   789    40         11     0.0160 Iter7  

lgbm_final <- 
  boost_tree(
    trees = 300, 
    learn_rate = best_params$learn_rate,
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

best_params_manual <- 
  data.frame(
    mtry = 3848,
    min_n = 27,
    tree_depth = 2,
    learn_rate = 0.0904 
  )

best_model_manual <- 
  boost_tree(
    trees = 300, 
    learn_rate = best_params_manual$learn_rate,
    tree_depth = best_params_manual$tree_depth, 
    min_n = best_params_manual$min_n,
    loss_reduction = 0, 
    mtry = best_params_manual$mtry
  ) %>%
  set_engine(engine = "lightgbm") %>%
  set_mode(mode = "classification") %>% 
  translate()


# Once the final model is determined, the entire training set is used for the last fit.
testing <- last_fit(best_model_manual, full_dataset, split = dataset_split) 

testing %>% 
  collect_metrics()


#Estas metricas son para una BO optimizada por RECALL, us 1
# .metric  .estimator .estimate .config             
# <chr>    <chr>          <dbl> <chr>               
#   1 accuracy binary         0.966 Preprocessor1_Model1
# 2 roc_auc  binary         0.675 Preprocessor1_Model1

#Estas metricas son para una BO optimizada por RECALL, us 2
# .metric  .estimator .estimate .config             
# <chr>    <chr>          <dbl> <chr>               
#   1 accuracy binary         0.966 Preprocessor1_Model1
# 2 roc_auc  binary         0.708 Preprocessor1_Model1

#Estas metricas son para una BO optimizada por RECALL, us 15
# .metric  .estimator .estimate .config             
# <chr>    <chr>          <dbl> <chr>               
#   1 accuracy binary         0.966 Preprocessor1_Model1
# 2 roc_auc  binary         0.668 Preprocessor1_Model1


#Estas metricas son para una BO optimizada por RECALL
#   1 accuracy binary         0.969 Preprocessor1_Model1
# 2 roc_auc  binary         0.642 Preprocessor1_Model1

#Estas metricas son para una BO optimizada por PR_AUC, undersampling 2
# .metric  .estimator .estimate .config             
# <chr>    <chr>          <dbl> <chr>               
#   1 accuracy binary         0.966 Preprocessor1_Model1
# 2 roc_auc  binary         0.631 Preprocessor1_Model1

#Estas metricas son para una BO optimizada por PR_AUC
# 1 accuracy binary         0.966 Preprocessor1_Model1
# 2 roc_auc  binary         0.705 Preprocessor1_Model1

predicciones <- 
  testing %>% 
  collect_predictions() 
# Optimizando por recall us1 levanta 1 AS 1 TP y otro FP
# Optimizando por recall us2 levanta 2 AS 1 correctamente
# Optimizando por recall us15 levanta 1 AS 1 TP y otro FP
# Optimizando por recall levanta 2 AS 1 correctamente
# Optimizando por pr_auc undersampling 2 levanta 0 AS 1

# Optimizando por pr_auc levanta 0 AS 1

# cm <- testing %>% #Esto no funciona
#  conf_mat(obs, pred)

#autoplot(cm, type = "heatmap")

df_predicciones <- as.data.frame(testing$.predictions)

violin_plot = ggplot(df_predicciones, aes(x=AS, y=.pred_1, group=AS, fill=factor(AS))) + 
  geom_violin() +
  theme_bw() +
  guides(scale="none", fill=guide_legend(title="AS"))

violin_plot

# Modelo final----

# El fit en todo el dataset de entrenamiento esta en testing

final_lgbm_model <- 
  boost_tree(
    trees = 300, 
    learn_rate = best_params_manual$learn_rate,
    tree_depth = best_params_manual$tree_depth, 
    min_n = best_params_manual$min_n,
    loss_reduction = 0, 
    mtry = best_params_manual$mtry
  ) %>%
  set_engine(engine = "lightgbm") %>%
  set_mode(mode = "classification")

final_wflow <- 
  workflow() %>% 
  add_formula(AS ~ .) %>% 
  add_model(final_lgbm_model) 

final_fit <- final_wflow %>%  fit(data = train_data_total)


# Feature importance----
simple_explainer <- 
  explain_tidymodels(
    final_fit,
    data = dplyr::select(train_data_total, -AS),
    y = as.integer(train_data_total$AS),
    label = "simple lgbm explainer",
    verbose = FALSE
)

simple_global_explainer <- 
  simple_explainer %>% 
  model_parts() #Esta función genera explicaciones globales

# Esto no funciona y no entiendo por qué
# Es este error https://www.statology.org/contrasts-applied-to-factors-with-2-or-more-levels/
# Pero veo mas de 3 levels en todas las variables
simple_breakdown <- predict_parts(
  explainer = simple_explainer, 
  new_observation = test_data[12,] %>%  select(-AS))

# Función tomada de https://www.tmwr.org/explain.html
ggplot_imp <- function(...) {
  obj <- list(...)
  metric_name <- attr(obj[[1]], "loss_name")
  metric_lab <- paste(metric_name, 
                      "after permutations\n(higher indicates more important)")
  
  full_vip <- bind_rows(obj) %>%
    filter(variable != "_baseline_")
  
  perm_vals <- full_vip %>% 
    filter(variable == "_full_model_") %>% 
    group_by(label) %>% 
    summarise(dropout_loss = mean(dropout_loss))
  
  p <- full_vip %>%
    filter(variable != "_full_model_") %>% 
    mutate(variable = fct_reorder(variable, dropout_loss)) %>%
    ggplot(aes(dropout_loss, variable)) 
  if(length(obj) > 1) {
    p <- p + 
      facet_wrap(vars(label)) +
      geom_vline(data = perm_vals, aes(xintercept = dropout_loss, color = label),
                 size = 1.4, lty = 2, alpha = 0.7) +
      geom_boxplot(aes(color = label, fill = label), alpha = 0.2)
  } else {
    p <- p + 
      geom_vline(data = perm_vals, aes(xintercept = dropout_loss),
                 size = 1.4, lty = 2, alpha = 0.7) +
      geom_boxplot(fill = "#91CBD765", alpha = 0.4)
    
  }
  p +
    theme(legend.position = "none") +
    labs(x = metric_lab, 
         y = NULL,  fill = NULL,  color = NULL)
}

ggplot_imp(simple_global_explainer)





# SHAP----
# https://community.rstudio.com/t/shap-values-with-tidymodels/147217
# https://www.tmwr.org/explain.html

test_case <- test_data[1,]

shap_values <-  
  predict_parts(
    explainer = simple_explainer, 
    new_observation = test_case, 
    type = "shap",
    B = 20
  )

