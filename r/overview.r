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
path = "cluster_results/plots/"

## Load data

table = read.csv(paste(path,"output.csv", sep=""))
table <- table %>% rename(departments.SDP = X.departments,
                 points.SDP = X.ND.Points,
                 time.SDP = Total.Time)

table_arezoo = read.csv(paste("cluster_results/Arezoo/logs/output.csv", sep=""))
table_arezoo <- data.frame(table_arezoo)
table_arezoo <- dplyr::filter(table_arezoo, !grepl("Instance", Instance))
table_arezoo$STAH <- "STAT"
table_arezoo$Instance <- sub(".", "", table_arezoo$Instance)
table_arezoo$X.departments <- as.numeric(sub(".", "", table_arezoo$X.departments))
table_arezoo$X.ND.Points <- as.numeric(sub(".", "", table_arezoo$X.ND.Points))
table_arezoo$t.MIP <- as.numeric(sub(".", "", table_arezoo$t.MIP))
table_arezoo <- table_arezoo %>% rename(departments.LP = `X.departments`,
                                        points.LP = X.ND.Points,
                                        time.LP = t.MIP)

results <- merge(table, table_arezoo, by="Instance")
view(results)


table$Instance[1] == table_arezoo$Instance[9]

plot(table$X.departments, table$X.ND.Points)
plot(table$X.departments, table$Total.Time)

#ggplot(table, aes(Total.Time, colour = prelift, shape = prelift, linetype = prelift)) +
ggplot(table, aes(Total.Time)) +
  stat_ecdf(geom = "point", pad = FALSE) +
  stat_ecdf(geom = "step", pad = FALSE) + 
  scale_y_continuous(limits = c(0,1), labels = c("0", "25", "50", "75", "100")) +
  xlab("runtime [s]") + 
  ylab("#instances [%]") +
  labs(colour = "Setting", shape = "Setting", linetype = "Setting") +
  theme(text = element_text(family = "Helvetica"))


ggplot(table %>% filter(Total.Time < 3600), aes(X.departments, X.ND.Points)) +
    geom_point() +
    geom_smooth() +
    xlab("departments") + 
    ylab("ND.Points") +
    theme(text = element_text(family = "Helvetica"))