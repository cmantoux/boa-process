library(ggplot2)

theme_set(
  theme_classic() + 
    theme(legend.position = "right")
)

data = read.csv(file = 'prob_crit_H.csv', header = TRUE, dec = '.', sep = ',')

fig = ggplot(data) + 
  aes(x = H, y = Proba_critique) + 
  geom_point() +
  labs(x = 'Maximal dormancy duration H', y = bquote('Critical patch extinction probability'~p[c](H)))

fig = fig + ylim(0,1)

fig = fig + scale_x_continuous(expand = c(0,0), breaks = c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), limits = c(0,10.1))

fig = fig + scale_y_continuous(expand = c(0,0), limits = c(0,1))

fig = fig + geom_line(size = 0.5, colour = 'black')

fig = fig + geom_hline(aes(yintercept = 0.46), color = "red", linetype = "dashed", size = 1)

fig = fig + geom_hline(aes(yintercept = 1), color = "black",size = 0.25)

fig = fig + theme(axis.text.x = element_text(size = 14), axis.text.y = element_text(size = 14))
fig = fig + theme(legend.text = element_text(size = 14), legend.title = element_text(size = 16))
fig = fig + theme(axis.title = element_text(size = 16))

fig