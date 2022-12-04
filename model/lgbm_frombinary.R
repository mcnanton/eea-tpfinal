library(lightgbm)
library(dplyr)
library(tidymodels)
library(shapviz)
library(caret)

set.seed(1234)
options(scipen=999) #desactivamos notación científica

# Cargo data y creo train y test
dataset <- read.delim("./datasets/from_binary.txt",
                      sep = ",")

dataset_split <- initial_split(dataset, prop = 0.80, strata = AS) #Partición 80%

train_data_total <- training(dataset_split) 
test_data <- testing(dataset_split) 

X_test_matrix <- test_data %>% 
  select(-AS,
         -AGI) %>% 
  as.matrix()

X_matrix <- train_data_total %>% 
  select(-AS,
         -AGI) %>% 
  as.matrix()


dtrain <- lgb.Dataset(X_matrix, 
                      label = as.numeric(train_data_total$AS))

# Usamos los hiperparametros que surgieron de una optimizacion bayesiana
# en el script modelo_frombinary.R
params <- list(
  objective = "binary",
  num_iterations = 834,
  feature_fraction = 0.31,
  min_data_in_leaf = 7,
  max_depth = 3,
  learning_rate  = 0.0465
)


# Entrenamos el modelo
model <- lgb.train(
  params,
  data = dtrain, 
  verbose = -1L
)

# Importancia de cada feature en el modelo
tree_imp1 <- lgb.importance(model, percentage = TRUE)
tree_imp2 <- lgb.importance(model, percentage = FALSE)

# predeciomos test
prediccion <- predict(model,
                      data = X_test_matrix)

df_predicciones <- test_data %>% 
  mutate(AS_predicha = prediccion,
         AS_real = test_data$AS) %>% 
  select(AS_predicha,
         AS_real)

confusionMatrix(factor(df_predicciones$AS_predicha), 
                factor(df_predicciones$AS_real))


# vemos contribucion de cada feature
predict_contrib <- predict(model,
                           data=    X_test_matrix, 
                           predcontrib = TRUE)

df_shap_values <- as.data.frame(predict_contrib) 

test_cols <- colnames(X_test_matrix)
colnames(df_shap_values) <- test_cols

# unimos predicción con contribucion
full_values <- df_predicciones %>% 
  bind_cols(df_shap_values)


# Analizo SHAP values
# Genero obejto SHAP
shp <- shapviz(model, X_test_matrix, X = X_test_matrix)

# interpretabilidad local
wf_2406 <-sv_waterfall(shp, row_id = 524)
wf_524 <-sv_waterfall(shp, row_id = 524)

# interpretabilidad global
# promedio del modulo de importancia local
importance_plot <- sv_importance(shp)

# Beeswarm plot
beeswarm_plot <- sv_importance(shp, kind = "beeswarm")

# dependence
sv_dependence(shp, v = "TTGTTT", "auto")

