library(ggplot2)
library(multcomp)
library(car)

myData = read.csv(file = 'smd_proba_ext.csv', header = TRUE, dec = '.', sep  = ',')

#Generation figure a)
a = ggplot(myData, aes(x = species, y = ecart_relatif_smd_p_ext))
a = a + geom_boxplot(alpha = 1)
a = a + theme(axis.text.x = element_text(angle = 90, size = 14), axis.text.y = element_text(size = 14))
a = a + theme(axis.title = element_text(size = 16))
a = a + labs(x = 'Species', y = 'Mean SMDs intra/inter streets')
a = a + guides(fill=FALSE)
a = a + geom_hline(aes(yintercept = 1), color = "red",size = 1)
a = a + scale_y_continuous(breaks = c(0, 1, 2, 4))

a

#Generation figure b)
a = ggplot(myData, aes(x = street, y = ecart_relatif_smd_p_ext))
a = a + geom_boxplot(alpha = 1)
a = a + theme(axis.text.x = element_text(angle = 90, size = 14), axis.text.y = element_text(size = 14))
a = a + theme(axis.title = element_text(size = 16))
a = a + labs(x = 'Street', y = 'Mean SMDs intra/inter streets')
a = a + guides(fill=FALSE)
a = a + geom_hline(aes(yintercept = 1), color = "red",size = 1)
a = a + scale_y_continuous(breaks = c(0, 1, 2, 4))

a