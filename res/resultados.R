
library(readxl)
library(tidyverse)
library(purrr)


#historiales sin fisica 
tabla <- read_excel('historiales_pinn.xlsx')


lista_dfs <- split(tabla, seq(nrow(tabla)))


df.1 <- lista_dfs[[1]]
df.2 <- lista_dfs[[2]]
df.3 <- lista_dfs[[3]]
df.4 <- lista_dfs[[4]]
df.5 <- lista_dfs[[5]]


#conversion
df_listas <- df.1 %>% 
  mutate(across(everything(), convertir_vector))


#ver cantidad de filas
map(df_listas, ~ length(.x[[1]]))


convertir_vector <- function(x) {
  lapply(x, function(s) as.numeric(strsplit(gsub("\\[|\\]", "", s), ",")[[1]]))
}


longitudes <- map_int(df_listas, ~ length(.x[[1]]))
min_len <- min(longitudes)


df_recortado <- df_listas %>%
  mutate(across(everything(), ~ lapply(.x, function(vec) vec[1:min_len]))) %>%
  unnest(cols = everything())


write.csv(df_recortado, 'df_1_final.csv', row.names = FALSE)




for (i in seq_along(lista_dfs)) {
  write.csv(lista_dfs[[i]], paste0("df_", i, ".csv"), row.names = FALSE)
}



convertir_vector <- function(x) {
  lapply(x, function(s) as.numeric(strsplit(gsub("\\[|\\]", "", s), ",")[[1]]))
}

df_expandido <- df.1 %>%
  mutate(across(everything(), convertir_vector)) %>%
  unnest(cols = everything())
                
                

#historiales con fisica 

tabla.f<- read.csv('historiales_con_Fisica.csv')

lista_dfs_f <- split(tabla.f, seq(nrow(tabla.f)))

for (i in seq_along(lista_dfs_f)) {
  write.csv(lista_dfs_f[[i]], paste0("df_f_", i, ".csv"), row.names = FALSE)
}


