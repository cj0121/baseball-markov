library(lme4)

# rm(list = ls())

getwd()
setwd("/Users/honeybunny/desktop/txt/sports/baseball-sim/PA/R")

df <- data.frame(read.csv('PA_data_merged_19_012223.csv'))

# (df$platoon)

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

hr.baseline.model <- glmer(HR ~ HR_proba + (1 + temp | HR_pf), data=df, family = binomial)
# Warning messages:
#   1: In checkConv(attr(opt, "derivs"), opt$par, ctrl = control$checkConv,  :
#                     Model failed to converge with max|grad| = 0.00372811 (tol = 0.002, component 1)
#                   2: In checkConv(attr(opt, "derivs"), opt$par, ctrl = control$checkConv,  :
#                                     Model is nearly unidentifiable: very large eigenvalue
#                                   - Rescale variables?;Model is nearly unidentifiable: large eigenvalue ratio
#                                   - Rescale variables? 


X1B.model_1 <- glmer(X1B ~ X1B_proba + (1|platoon) + (1|ump_id) + (1|temp), data=df, family = binomial)
# Warning: boundary (singular) fit: see help('isSingular')
proba.1B <- data.frame(fixef(X1B.model_1))
platoon.1B <- data.frame(ranef(X1B.model_1)[['platoon']])
umpire.1B <- data.frame(ranef(X1B.model_1)[['ump_id']])
temp.1B <- data.frame(ranef(X1B.model_1)[['temp']])

X2B.model_1 <- glmer(X2B ~ X2B_proba + (1|platoon) + (1|ump_id) + (1|temp), data=df, family = binomial)
# Warning: boundary (singular) fit: see help('isSingular')
proba.2B <- data.frame(fixef(X2B.model_1))
platoon.2B <- data.frame(ranef(X2B.model_1)[['platoon']])
umpire.2B <- data.frame(ranef(X2B.model_1)[['ump_id']])
temp.2B <- data.frame(ranef(X2B.model_1)[['temp']])

X3B.model_1 <- glmer(X3B ~ X3B_proba + (1|platoon) + (1|ump_id) + (1|temp), data=df, family = binomial)
# Warning: boundary (singular) fit: see help('isSingular')
proba.3B <- data.frame(fixef(X3B.model_1))
platoon.3B <- data.frame(ranef(X3B.model_1)[['platoon']])
umpire.3B <- data.frame(ranef(X3B.model_1)[['ump_id']])
temp.3B <- data.frame(ranef(X3B.model_1)[['temp']])

HR.model_1 <- glmer(HR ~ HR_proba + (1|platoon) + (1|ump_id) + (1|temp)+ (1|sprint_speed), data=df, family = binomial)
proba.HR <- data.frame(fixef(HR.model_1))
platoon.HR <- data.frame(ranef(HR.model_1)[['platoon']])
umpire.HR <- data.frame(ranef(HR.model_1)[['ump_id']])
temp.HR <- data.frame(ranef(HR.model_1)[['temp']])
speed.HR <- data.frame(ranef(HR.model_1)[['sprint_speed']])

FO.model_1 <- glmer(FO ~ FO_proba + (1|platoon) + (1|ump_id) + (1|temp), data=df, family = binomial)
# Warning: boundary (singular) fit: see help('isSingular')
proba.FO <- data.frame(fixef(FO.model_1))
platoon.FO <- data.frame(ranef(FO.model_1)[['platoon']])
umpire.FO <- data.frame(ranef(FO.model_1)[['ump_id']])
temp.FO <- data.frame(ranef(FO.model_1)[['temp']])

BB.model_1 <- glmer(BB ~ BB_proba + (1|platoon) + (1|ump_id) + (1|temp), data=df, family = binomial)
proba.BB <- data.frame(fixef(BB.model_1))
platoon.BB <- data.frame(ranef(BB.model_1)[['platoon']])
umpire.BB <- data.frame(ranef(BB.model_1)[['ump_id']])
temp.BB <- data.frame(ranef(BB.model_1)[['temp']])

K.model_1 <- glmer(K ~ K_proba + (1|platoon) + (1|ump_id) + (1|temp), data=df, family = binomial)
proba.K <- data.frame(fixef(K.model_1))
platoon.K <- data.frame(ranef(K.model_1)[['platoon']])
umpire.K <- data.frame(ranef(K.model_1)[['ump_id']])
temp.K <- data.frame(ranef(K.model_1)[['temp']])

other.model_1 <- glmer(other ~ other_proba + (1|platoon) + (1|ump_id) + (1|temp), data=df, family = binomial)
proba.other <- data.frame(fixef(other.model_1))
platoon.other <- data.frame(ranef(other.model_1)[['platoon']])
umpire.other <- data.frame(ranef(other.model_1)[['ump_id']])
temp.other <- data.frame(ranef(other.model_1)[['temp']])

df.platoon <- do.call("cbind", list(platoon.1B, platoon.2B, platoon.3B, platoon.HR, platoon.FO, platoon.K, platoon.BB, platoon.other))
colnames(df.platoon) <- c("1B","2B","3B","HR","FO","K","BB","other")

df.umpire <- do.call("cbind", list(umpire.1B, umpire.2B, umpire.3B, umpire.HR, umpire.FO, umpire.K, umpire.BB, umpire.other))
colnames(df.umpire) <- c("1B","2B","3B","HR","FO","K","BB","other")

df.temp <- do.call("cbind", list(temp.1B, temp.2B, temp.3B, temp.HR, temp.FO, temp.K, temp.BB, temp.other))
colnames(df.temp) <- c("1B","2B","3B","HR","FO","K","BB","other")

df.speed <- do.call("cbind", list(speed.HR))
colnames(df.speed) <- c("HOMERUN")

df.proba <- do.call("cbind", list(proba.1B, proba.2B, proba.3B, proba.HR, proba.FO, proba.K, proba.BB, proba.other))
colnames(df.proba) <- c("1B","2B","3B","HR","FO","K","BB","other")

# write.csv(df.platoon, "coefficients/platoon.csv")
# write.csv(df.umpire, "coefficients/umpire.csv")
# write.csv(df.temp, "coefficients/temp.csv")
# write.csv(df.speed, "coefficients/speed.csv")
# write.csv(df.proba, "coefficients/proba.csv")
