rm(list=ls())
tabX=read.csv("extinction_risk_with_traits.csv",header=T,sep=",",check.names = T)

library(rstatix)
library(FSA)
library(ggplot2)
library(dplyr)

#Global extinction risk
###Flowering period
fig = ggplot(tabX) + 
  aes(x = PeriodeF, y = Risquemoyen) + 
  geom_boxplot(outlier.shape = NA) +
  labs(x = "", y = "Average MaxGER") +
  ylim(0,1) +
  scale_x_discrete(limits=c("short","medium","long"))

fig = fig + theme(axis.text.x = element_text(size = 14), axis.text.y = element_text(size = 14))
fig = fig + theme(legend.text = element_text(size = 14), legend.title = element_text(size = 16))
fig = fig + theme(axis.title = element_text(size = 16))

fig

###Maximal height
fig = ggplot(tabX) + 
  aes(x = Maximum_height, y = Risquemoyen) + 
  geom_point()  + 
  labs(x = "", y = "Average MaxGER") + ylim(0,1)

fig = fig + theme(axis.text.x = element_text(size = 14), axis.text.y = element_text(size = 14))
fig = fig + theme(legend.text = element_text(size = 14), legend.title = element_text(size = 16))
fig = fig + theme(axis.title = element_text(size = 16))

fig

###Beginning of flowering period
tabX$PeriodeF = with(tabX, reorder(debf2,Risquemoyen,mean))

fig = ggplot(tabX) + 
  aes(x = debf2, y = Risquemoyen) + 
  geom_boxplot(outlier.shape = NA) +
  labs(x = "", y = "Average MaxGER") +
  ylim(0,1) 

fig = fig + theme(axis.text.x = element_text(size = 14), axis.text.y = element_text(size = 14))
fig = fig + theme(legend.text = element_text(size = 14), legend.title = element_text(size = 16))
fig = fig + theme(axis.title = element_text(size = 16))

fig

#Local extinction risk
kruskal.test(Risquemoyenlocal~Dissemination,data=tabX)
kruskal.test(Risquemoyenlocal~PeriodeF,data=tabX)
cor.test(tabX$Massegraine,tabX$Risquemoyenlocal,method="spearman")
wilcox_test(Risquemoyenlocal~Resistance_chaleur,data=tabX,alternative="two.sided")
kruskal.test(Risquemoyenlocal~poll_vect,data=tabX)
cor.test(tabX$Maximum_height,tabX$Risquemoyenlocal,method="spearman")
wilcox_test(Risquemoyenlocal~debf2,data=tabX,alternative="two.sided")

###Maximal height
fig = ggplot(tabX) + 
  aes(x = Maximum_height, y = Risquemoyenlocal) + 
  geom_point()  + 
  labs(x = "", y = "Average LER") + ylim(0,1)

fig = fig + theme(axis.text.x = element_text(size = 14), axis.text.y = element_text(size = 14))
fig = fig + theme(legend.text = element_text(size = 14), legend.title = element_text(size = 16))
fig = fig + theme(axis.title = element_text(size = 16))

fig

###Beginning of flowering period
tabX$PeriodeF = with(tabX, reorder(debf2,Risquemoyenlocal,mean))

fig = ggplot(tabX) + 
  aes(x = debf2, y = Risquemoyenlocal) + 
  geom_boxplot(outlier.shape = NA) +
  labs(x = "", y = "Average LER") +
  ylim(0,1) 

fig = fig + theme(axis.text.x = element_text(size = 14), axis.text.y = element_text(size = 14))
fig = fig + theme(legend.text = element_text(size = 14), legend.title = element_text(size = 16))
fig = fig + theme(axis.title = element_text(size = 16))

fig


