rm(list=ls())
tabX=read.csv(file = "extinction_risk_with_traits.csv",header=T,sep=",")

library(rstatix)
library(FSA)
library(ggplot2)
library(dplyr)

#Global extinction risk
##Dispersal mechanism
kruskal.test(Risquemoyen~Dissemination,data=tabX)

##Flowering duration
kruskal.test(Risquemoyen~PeriodeF,data=tabX)
dunnTest(Risquemoyen ~ PeriodeF,data=tabX,method="holm") #Kruskal test was significant

##Seed mass
cor.test(tabX$Massegraine,tabX$Risquemoyen,method="spearman")

##Heat preference
wilcox_test(Risquemoyen~Resistance_chaleur,data=tabX,alternative="two.sided")

##Pollination vector
kruskal.test(Risquemoyen~poll_vect,data=tabX)

##Maximal height
cor.test(tabX$Maximum_height,tabX$Risquemoyen,method="spearman")

##Beginning of flowering period
wilcox_test(Risquemoyen~debf2,data=tabX,alternative="two.sided")

#Local extinction risk
##Dispersal mechanism
kruskal.test(Risquemoyenlocal~Dissemination,data=tabX)

##Flowering duration
kruskal.test(Risquemoyenlocal~PeriodeF,data=tabX)

##Seed mass
cor.test(tabX$Massegraine,tabX$Risquemoyenlocal,method="spearman")

##Heat preference
wilcox_test(Risquemoyenlocal~Resistance_chaleur,data=tabX,alternative="two.sided")

##Pollination vector
kruskal.test(Risquemoyenlocal~poll_vect,data=tabX)

##Maximal height
cor.test(tabX$Maximum_height,tabX$Risquemoyenlocal,method="spearman")

##Beginning of flowering period
wilcox_test(Risquemoyenlocal~debf2,data=tabX,alternative="two.sided")
