
library(readxl)


#historiales sin fisica 
tabla <- read_excel('historiales_pinn.xlsx')


lista_dfs <- split(tabla, seq(nrow(tabla)))


df.1 <- lista_dfs[[1]]
df.2 <- lista_dfs[[2]]
df.3 <- lista_dfs[[3]]
df.4 <- lista_dfs[[4]]
df.5 <- lista_dfs[[5]]


for (i in seq_along(lista_dfs)) {
  write.csv(lista_dfs[[i]], paste0("df_", i, ".csv"), row.names = FALSE)
}


#historiales con fisica 

tabla.f<- read.csv('historiales_con_Fisica.csv')

lista_dfs_f <- split(tabla.f, seq(nrow(tabla.f)))

for (i in seq_along(lista_dfs_f)) {
  write.csv(lista_dfs_f[[i]], paste0("df_f_", i, ".csv"), row.names = FALSE)
}


