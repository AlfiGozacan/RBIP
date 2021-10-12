setwd("C:/Users/agozacan/OneDrive - Humberside Fire and Rescue Service/RBIP Project/EPC Data")
df <- read.csv("combined_epc_data.csv")

names(df)

df2 <- data.frame("band" = unique(df$ASSET_RATING_BAND))

for(i in 1:length(df2$band)){
  df2$freq[i] <- sum(df$ASSET_RATING_BAND == df2$band[i])
}

df2 <- df2[-c(8, 9),]

library(ggplot2)

ggplot(data = df2, mapping = aes(x = band, y = freq)) +
  geom_col(aes(fill = ..x..)) +
  scale_fill_gradient(guide = "none", low = "green", high = "red") +
  labs(x = "EPC Rating",
       y = "Frequency",
       title = "") +
  theme(text = element_text(size = 12,
                            family = "mono"),
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank()) +
  geom_label(aes(label = band), nudge_y = 1)

ggplot(data = df, mapping = aes(x = log(FLOOR_AREA))) +
  geom_histogram(binwidth = 0.1, fill = "royalblue1") +
  labs(x = "Logarithm of Floor Area",
       y = "Frequency",
       title = "") +
  theme(text = element_text(size = 12,
                            family = "mono"))

ggplot(data = df[-which(df$ASSET_RATING_BAND %in% c("A+", "INVALID!")),], mapping = aes(x = ASSET_RATING_BAND, y = log(FLOOR_AREA))) +
  geom_violin(fill = "lightblue", trim = FALSE) +
  stat_summary(fun.data = "mean_cl_boot",
               geom = "pointrange",
               colour = "red") +
  labs(x = "EPC Rating",
       y = "Logarithm of Floor Area",
       title = "") +
  theme(text = element_text(size = 12,
                            family = "mono"))
