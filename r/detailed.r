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
path = "plots"

## Load data

getwd()
filename = "AC_12_70_5_AC_12_30_5_detailed.csv"

table = read.csv(paste(path,filename, sep="/"))

table$Rootgap.OPT = table$Rootgap.OPT*100

plot(table$BNB.Nodes, table$Rootgap.OPT, type = )
