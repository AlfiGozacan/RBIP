setwd("C:/Users/agozacan/OneDrive - Humberside Fire and Rescue Service/RBIP Project/Merged Data")
df <- read.csv("rbip_epc_irs_join.csv")

model_cols = c("FSEC_DESCRIPT", "GRS", "FIRE.SAFETY.STATUS",
              "Satisfactory", "ASSET_RATING", "ASSET_RATING_BAND", "MAIN_HEATING_FUEL", "inc.2010", "inc.2011", "inc.2012", "inc.2013",
              "inc.2014", "inc.2015", "inc.2016", "inc.2017",
              "inc.2018", "inc.2019", "inc.2020")

df = df[, model_cols]

categorical_cols = c("FSEC_DESCRIPT", "FIRE.SAFETY.STATUS", "ASSET_RATING_BAND",
                    "Satisfactory", "MAIN_HEATING_FUEL")

for(col in categorical_cols){
  df[, col] <- as.factor(df[, col])
}

df[is.na(df)] = 0

df$inc.2020[df$inc.2020 > 0] = 1

training_indices = sample(1:length(df$inc.2020), round(0.7 * length(df$inc.2020)), replace=FALSE)

training_data = df[training_indices,]
test_data = df[-training_indices,]

logreg <- glm(inc.2020 ~ ., data = training_data, family = binomial(link = "logit"))

# logreg
# summary(logreg)
# anova(logreg)

probabilities <- predict(logreg, newdata = test_data, type = "response")

hist(probabilities, nclass=20)

#### PLOT ####

test_data$PREDICTION <- probabilities

library(pROC)

roc(inc.2020 ~ PREDICTION, data = test_data, plot = TRUE)

#### CONFUSION MATRIX ####

for(i in 1:length(test_data$inc.2020)){
  
  test_data$PREDICTION_BINARY[i] = round(test_data$PREDICTION[i])
  
}

library(caret)

confusionMatrix(as.factor(test_data$inc.2020), as.factor(test_data$PREDICTION_BINARY), positive = "1")
