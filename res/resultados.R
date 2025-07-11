
library(readxl)

tabla <- read_excel('historiales_pinn.xlsx')



lista_dfs <- split(tabla, seq(nrow(tabla)))


df.1 <- lista_dfs[[1]]
df.2 <- lista_dfs[[2]]
df.3 <- lista_dfs[[3]]
df.4 <- lista_dfs[[4]]
df.5 <- lista_dfs[[5]]
