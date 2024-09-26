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

hgd()
hgd_browse()

## File paths
path = "cluster_results/stats/"
filename = "AC_8_30_5_AC_8_30_5_1_detailed.txt"
## Load data

table = read.csv(paste(path,filename, sep=""))