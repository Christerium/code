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
path = "cluster_results/stats"

hgd()
hgd_browse()

## Load data

getwd()

name = "AC_12_50_20_AC_12_50_20_1"
#name = "S11_SRFLP11"

filename = paste(name, "_detailed.txt", sep="")
filename2 = paste(name, ".csv", sep="")

t_det= read.csv(paste(path,filename, sep="/"))
t_det_a= read.csv(paste("cluster_results/Arezoo/output/", filename2, sep=""))

t_det$method <- "SDP-based"
t_det_a$method <- "IP"

t_det <- t_det %>% rename(  rootgap.OPT = Rootgap.OPT,
                            BNB = BNB.Nodes,
                            rootgap = Rootgap,
                            time = Total.Time)

t_det_a <- t_det_a %>% rename(  rootgap.OPT = Rootgap_OPT_MIP.,
                            BNB = BNB_Nodes_MIP,
                            rootgap = Rootgap_MIP.,
                            time = Total_Time_MIP.s.)

t_det$rootgap <- 100 * t_det$rootgap
t_det$rootgap.OPT <- t_det$rootgap.OPT*100

subset(t_det, select = -SDP.Time)

t <- rbind(subset(t_det, select = -SDP.Time), t_det_a)

ggplot(t, aes(OBJ1, rootgap.OPT, color = method)) +
  geom_point() +
  geom_smooth(se = FALSE) +
  xlab("f1") + 
  ylab("Root gap optimal (%)") +
  theme(text = element_text(family = "Helvetica"))
ggsave("rootgap_optimal.pdf", width = 5, height = 5)

ggplot(t, aes(OBJ1, BNB, color = method)) +
  geom_point() +
  geom_smooth(se = FALSE) +
  xlab("f1") + 
  ylab("# BNB Nodes") +
  theme(text = element_text(family = "Helvetica"))
ggsave("bnb_nodes.pdf", width = 5, height = 5)