# Setup----
library(dplyr)
library(tidymodels)
library(rsample)
library(bonsai)
library(DALEXtra)
library(themis)

tidymodels_prefer()

set.seed(1234)

# Particiones----
dataset <- read.delim("datasets/from_binary.txt",
                      sep = ",") %>% 
  mutate(AS = factor(AS, levels=c('1','0')))

dataset_split <- initial_split(dataset, prop = 0.80, strata = AS)
train_data_total <- training(dataset_split) #Partición 80%, no extraemos validación porque vamos a hacer cross-validation
test_data <- testing(dataset_split) 


train_data_formula <-
  recipe(AS ~ ., data = train_data_total) %>%
  step_downsample(AS, 
                  under_ratio = 2, # El doble de AS=0 respecto a AS=1
                  seed = 42)

# Hiperparámetros----

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
    metrics = metric_set(recall),
    control = ctrl
  )

# Modelo con mejores parámetros de BO
show_best(lgbm_bo)

autoplot(lgbm_bo, type = "performance")

best_params <-
  lgbm_bo %>% 
  select_best(metric = "recall")

best_params

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

# Testing ----
# Fiteamos el modelo en train_data_total y generamos predicción en test_data

full_dataset <- 
  recipe(AS ~ ., 
         data = dataset)

best_params_manual <- 
  data.frame(
    mtry = 1237,
    min_n = 834,
    tree_depth = 7,
    learn_rate = 0.0465 
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



predicciones <- 
  testing %>% 
  collect_predictions() 

# cm <- testing %>% #Esto no funciona
#  conf_mat(obs, pred)


df_predicciones <- as.data.frame(testing$.predictions)

violin_plot = ggplot(df_predicciones, aes(x=AS, y=.pred_1, group=AS, fill=factor(AS))) + 
  geom_violin() +
  theme_bw() +
  guides(scale="none", fill=guide_legend(title="AS"))

violin_plot

# Modelo final----

## El fit en todo el dataset de entrenamiento esta en testing
## Regeneramos el modelo (pese a que ya lo tenemos en last_fit() porque los explainers no se integran con esa función)


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

