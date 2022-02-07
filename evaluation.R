### Load libraries -------------

library(tidyverse)
library(gridExtra)

### Load data --------------------

df <- read.csv("C:/Users/agozacan/OneDrive - Humberside Fire and Rescue Service/RBIP Project/Input and Output/output.csv")

### Using the quartiled data -------------------

df %>%
  filter(inc.2020.bool == 1) %>%
  
  ggplot(data=., mapping=aes(x=QUARTILE)) +
  geom_histogram(binwidth=.5, fill="aquamarine3") +
  labs(
    title = "Quartile breakdown of all RBIP commercial properties where an incident occurred in 2020",
    x = "Quartile",
    y = "Frequency"
  )

years = c("inc.2010", "inc.2011", "inc.2012", "inc.2013", "inc.2014", "inc.2015", "inc.2016", "inc.2017", "inc.2018", "inc.2019")

for(j in 1:length(years)){
  
  varName <- paste("p", j, sep = "")
  
  index = grep(years[j], names(df))
  
  assign(varName, df %>%
           filter(df[,index] > 0) %>%
           ggplot(data=., mapping=aes(x=QUARTILE)) +
           geom_histogram(binwidth=.5, fill="aquamarine3") +
           labs(
             title = paste(years[j], "> 0"),
             x = "Quartile",
             y = "Frequency"
           )
  )
  
}

grid.arrange(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, nrow=3, ncol=4)
