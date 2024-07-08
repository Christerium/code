## Create tables for the MOSRFLP

## Libraries
library(tidyverse)
library(ggplot2)
library(xtable)
library(scales)
library(extrafont)
library(httpgd)
library(dplyr)
library(tidyr)
library(kableExtra)
library(knitr)
loadfonts(device = "postscript")

## File paths
path = "plots/stat/"

## Load data

table = read.csv(paste(path,"output.csv", sep=""))

plot(table$X.departments, table$X.ND.Points)
plot(table$X.departments, table$Total.Time)