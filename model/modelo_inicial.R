# Setup
library(dplyr)
library(tidymodels)
library(rsample)
tidymodels_prefer()

# Levantamos particiones
train_data = read.csv("datasets/train_data.csv")
val_data = read.csv("datasets/val_data.csv")
test_data = read.csv("datasets/test_data.csv")
