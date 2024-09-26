## Create frontiers for the MOSRFLP

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
table <- table %>% rename(departments = X.departments,
                 points = X.ND.Points,
                 time = Total.Time)
table$method <- "SDP-based"

table_arezoo = read.csv(paste("cluster_results/Arezoo/logs/output.csv", sep=""))
# table_arezoo <- data.frame(table_arezoo)
# table_arezoo <- dplyr::filter(table_arezoo, !grepl("Instance", Instance))
# table_arezoo$STAH <- "STAT"
table_arezoo$Instance <- sub(".", "", table_arezoo$Instance)
#table_arezoo$X.departments <- sub(".", "", table_arezoo$X.departments)
#table_arezoo$X.ND.Points <- as.numeric(sub(".", "", table_arezoo$X.ND.Points))
#table_arezoo$t.MIP <- as.numeric(sub(".", "", table_arezoo$t.MIP))
table_arezoo <- table_arezoo %>% rename(departments = `X.departments`,
                                        points = X.ND.Points,
                                        time = t.MIP)
table_arezoo$method <- "IP"
table_arezoo$time[table_arezoo$time >= 3595] = 3601

results <- rbind(table, table_arezoo)
results$method <- as.factor(results$method)
#view(results)

ggplot(results %>% filter(time < 3600), aes(departments, points, color = method)) +
  geom_point() +
  geom_smooth(se = FALSE) +
  xlab("departments") + 
  ylab("Non-dominated points") +
  theme(text = element_text(family = "Helvetica"))

ggplot(results %>% filter(time < 3600 & method =="SDP-based"), aes(departments, points)) +
  geom_point() +
  geom_smooth(se = FALSE) +
  xlab("departments") + 
  ylab("Non-dominated points") +
  theme(text = element_text(family = "Helvetica"))


plot(table$X.departments, table$X.ND.Points)
plot(table$X.departments, table$Total.Time)

#ggplot(table, aes(Total.Time, colour = prelift, shape = prelift, linetype = prelift)) +
ggplot(results, aes(time, color=method, points=method, linetype=method)) +
  stat_ecdf(geom = "point", pad = FALSE) +
  stat_ecdf(geom = "step", pad = FALSE) + 
  scale_y_continuous(limits = c(0,1), labels = c("0", "25", "50", "75", "100")) +
  xlab("runtime [s]") + 
  ylab("#instances [%]") +
  labs(colour = "Method", shape = "Method", linetype = "Method") +
  theme(text = element_text(family = "Helvetica"))
ggsave("runtime.pdf", width = 6, height = 4)

ggplot(table %>% filter(Total.Time < 3600), aes(X.departments, X.ND.Points)) +
    geom_point() +
    geom_smooth() +
    xlab("departments") + 
    ylab("ND.Points") +
    theme(text = element_text(family = "Helvetica"))


results2 <- merge(table, table_arezoo, by = c("Instance", "departments"))
results2 <- subset(results2, select=-c(STAH.x, STAH.y, method.x, method.y, points.x))
results2 <- results2[order(results2$departments), ]
results2 <- results2 %>%
  rename("SDP-based" = time.x, "IP" = time.y, "n" = departments, "ND Points" = points.y)


format_values <- function(sdp, ip) {
  if (sdp > 3600) {
    sdp_str <- "TL"
  } else {
    sdp_str <- sprintf("%.2f", sdp)
  }
  
  if (ip > 3600) {
    ip_str <- "TL"
  } else {
    ip_str <- sprintf("%.2f", ip)
  }
  
  if (sdp < ip) {
    sdp_str <- paste0("\\textbf{", sdp_str, "}")
  } else if (ip < sdp) {
    ip_str <- paste0("\\textbf{", ip_str, "}")
  }
  
  return(c(sdp_str, ip_str))
}

# Apply the custom function to the results2 data frame
results2$SDP_based_formatted <- mapply(format_values, results2$`SDP-based`, results2$IP)[1, ]
results2$IP_formatted <- mapply(format_values, results2$`SDP-based`, results2$IP)[2, ]

results2_latex <- results2 %>%
  select(Instance, n, `ND Points`, SDP_based_formatted, IP_formatted)

# Rename the columns for the LaTeX table
colnames(results2_latex) <- c("Instance", "n", "ND Points", "SDP-based", "IP")

# Generate the LaTeX code using xtable
latex_table <- xtable(results2_latex, caption = "Results Table", label = "tab:results")

sanitize_underscore <- function(x) {
  gsub("_", "\\\\_", x)
}

# Print the LaTeX code
print(latex_table, include.rownames = FALSE, sanitize.text.function = sanitize_underscore)