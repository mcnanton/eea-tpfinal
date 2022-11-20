# Setup
library(dplyr)
library(tidymodels)
library(rsample)
tidymodels_prefer()

set.seed(123)

dataset <- read.delim("datasets/WT_red_as_kmer.txt",
                      sep = ",")

# Veo que la variable AS es sumamente desbalanceada

table(dataset$AS)
# 0     1 
# 17514    89

# Split 80/20 conservando la proporción de dataset$AS
dataset_split <- initial_split(dataset, prop = 0.80, strata = AS)
train_data_total <- training(dataset_split)
test_data <- testing(dataset_split)

table(train_data_total$AS)
# 0     1 
# 14012    70 

table(test_data$AS)
# 0    1 
# 3502   19 

val_split <- initial_split(train_data_total, prop = 0.80, strata = AS)
train_data <- training(val_split)
val_data <- testing(val_split)

table(train_data$AS)
# 0     1 
# 11209    56 
table(val_data$AS)
# 0    1 
# 2803   14 

write.csv(test_data, "datasets/test_data.csv")
write.csv(train_data, "datasets/train_data.csv")
write.csv(train_data_total, "datasets/training_data.csv")
write.csv(val_data, "datasets/val_data.csv")

