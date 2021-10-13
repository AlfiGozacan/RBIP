setwd("C:/Users/agozacan/OneDrive - Humberside Fire and Rescue Service/RBIP Project/Merged Data")

df1 <- read.csv("epc_inc_left_join.csv")

df2 <- read.csv("fsa_inc_left_join.csv")

library(ggplot2)

ggplot(data = df1[-which(df1$inc.total > 10),], mapping = aes(x = as.factor(inc.total), y = log(FLOOR_AREA))) +
  geom_violin(fill = "lightblue", trim = FALSE) +
  stat_summary(fun.data = "mean_cl_boot",
               geom = "pointrange",
               colour = "red") +
  labs(x = "Number of Incidents",
       y = "Logarithm of Floor Area",
       title = "") +
  theme(text = element_text(size = 12,
                            family = "mono"))

ggplot(data = df2[-which(df2$inc.total > 10 | df2$RatingValue %in% c("Exempt", "AwaitingInspection")),], mapping = aes(x = as.factor(inc.total), y = RatingValue)) +
  geom_jitter() +
  labs(x = "Number of Incidents",
       y = "FSA Rating",
       title = "") +
  theme(text = element_text(size = 12,
                            family = "mono"))
