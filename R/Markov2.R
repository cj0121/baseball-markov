library(lme4)

setwd("/Users/honeybunny/desktop/txt/sports/baseball-sim/PA/R")

df <- data.frame(read.csv('Train_std_012523.csv'))

df$batter <- factor(df$batter)
df$pitcher <- factor(df$pitcher)
df$X1B <- factor(df$X1B)
df$X2B <- factor(df$X2B)
df$X3B <- factor(df$X3B)
df$HR <- factor(df$HR)
df$FO <- factor(df$FO)
df$K <- factor(df$K)
df$other <- factor(df$other)
df$home_team <- factor(df$home_team)
df$platoon <- factor(df$platoon)
df$ump_id <- factor(df$ump_id)


# model.1B = glmer(X1B ~ X1B_proba  + (1|platoon:home_team), data=df, family = "binomial") 
# # no warning
model.1B = glmer(X1B ~ X1B_proba  + (1|ump_id) + (1|platoon:home_team) + (1|temp_std:home_team), data=df, family = "binomial") 
# no warning

proba.1B <- data.frame(fixef(model.1B))
platoon.1B <- data.frame(ranef(model.1B)[['platoon']])
umpire.1B <- data.frame(ranef(model.1B)[['ump_id']])

model.2B <- glmer(X2B ~ X2B_proba + (1|ump_id) + (1|platoon:home_team) + (1|temp_std:home_team), data=df, family = "binomial")

model.3B <- glmer(X3B ~ X3B_proba + (1|platoon), data=df, family = "binomial")

model.HR <- glmer(HR ~ HR_proba + (1|ump_id) + (1|platoon:home_team) + (1|temp_std:home_team), data=df, family = "binomial")
