library(lightgbm)
library(dplyr)
library(tidymodels)

set.seed(1234)
options(scipen=999) #desactivamos notación científica

# Particiones----
dataset <- read.delim("datasets/from_binary.txt",
                      sep = ",")

dataset_split <- initial_split(dataset, prop = 0.80, strata = AS)
#Partición 80%, no extraemos validación porque vamos a hacer cross-validation
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

params <- list(
  objective = "binary",
  feature_fraction = 0.3,
  min_data_in_leaf = 834,
  max_depth = 7,
  learning_rate  = 0.0465
)

model <- lgb.train(
  params
  , data = dtrain
  , nrounds = 300L #A que corresponde en el fit de tidymodels
  , verbose = -1L
)

tree_imp1 <- lgb.importance(model, percentage = TRUE)
tree_imp2 <- lgb.importance(model, percentage = FALSE)

test_cols <- colnames(X_test_matrix)

prediccion <- predict(model,
                      data = X_test_matrix)

df_predicciones <- test_data %>% 
  mutate(AS_predicha = prediccion,
         AS_real = test_data$AS) %>% 
  select(AS_predicha,
         AS_real)

# Mandar CSV
# Devuelve (n_samples, n_classes, n_features + 1)
# https://stackoverflow.com/questions/58710152/lightgbm-predict-with-pred-contrib-true-for-multiclass-order-of-shap-values-in
predict_contrib <- predict(model,
                           data=    X_test_matrix, 
                           #type = "contrib" 
                           predcontrib = TRUE)

df_shap_values <- as.data.frame(predict_contrib) 
colnames(df_shap_values) <- test_cols

full_values <- df_predicciones %>% 
  bind_cols(df_shap_values)

write.csv(full_values, "preds_y_shap_values.csv")
write.csv(df_shap_values, "shap_values.csv")
