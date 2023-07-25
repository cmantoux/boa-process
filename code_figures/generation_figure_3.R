library(ggplot2)
library(multcomp)
library(car)

myData = read.csv(file = 'estimated_extinction_risk.csv', header = TRUE, dec = ',', sep  = ',')

#Generation Figure 3b
a = ggplot(myData, aes(x = reorder(species, risque_extinction_locale, FUN = median), y = risque_extinction_locale))
a = a + geom_boxplot(alpha = 1)
a = a + theme(axis.text.x = element_text(angle = 90, size = 12), axis.text.y = element_text(size = 12))
a = a + theme(axis.title = element_text(size = 14))
a = a + labs(x = 'Species', y = 'LER')
a = a + guides(fill=FALSE)

a

#Generation FIgure 3a
b = ggplot(myData, aes(x = reorder(species, risque_extinction_globale, FUN = median), y = risque_extinction_globale))
b = b + geom_boxplot(alpha = 1)
b = b + theme(axis.text.x = element_text(angle = 90, size = 12), axis.text.y = element_text(size = 12))
b = b + theme(axis.title = element_text(size = 14))
b = b + labs(x = 'Species', y = 'MaxGER')
b = b + geom_hline(aes(yintercept = 0.125), color = "red",size = 1)
b = b + guides(fill=FALSE)

b
