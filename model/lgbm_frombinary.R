library(lightgbm)
library(dplyr)
library(tidymodels)

set.seed(1234)

# Particiones----
dataset <- read.delim("datasets/from_binary.txt",
                      sep = ",")

dataset_split <- initial_split(dataset, prop = 0.80, strata = AS)
#Partición 80%, no extraemos validación porque vamos a hacer cross-validation
train_data_total <- training(dataset_split) 

X_matrix <- train_data_total %>% 
  select(-AS,
         -AGI) %>% 
  as.matrix()

test_data <- testing(dataset_split) 

dtrain <- lgb.Dataset(X_matrix, 
                      label = as.numeric(train_data_total$AS))

params <- list(
  objective = "binary",
  feature_fraction = 0.3,
  min_data_in_leaf = 834,
  max_depth = 7,
  learning_rate  = 0.0465
)

fit <- lgb.train(
  params
  , data = dtrain
  , nrounds = 300L
  , verbose = -1L
)

tree_imp1 <- lgb.importance(fit, percentage = TRUE)
tree_imp2 <- lgb.importance(fit, percentage = FALSE)
