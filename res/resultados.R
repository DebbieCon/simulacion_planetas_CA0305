
library(readxl)
library(tidyverse)
library(purrr)
library(dplyr)


#historiales sin fisica 
tabla <- read_excel('historiales_pinn.xlsx')


lista_dfs <- split(tabla, seq(nrow(tabla)))


df.1 <- lista_dfs[[1]]
df.2 <- lista_dfs[[2]]
df.3 <- lista_dfs[[3]]
df.4 <- lista_dfs[[4]]
df.5 <- lista_dfs[[5]]


convertir_vector <- function(x) {
  lapply(x, function(s) as.numeric(strsplit(gsub("\\[|\\]", "", s), ",")[[1]]))
}

#conversion
df_listas <- df.1 %>% 
  mutate(across(everything(), convertir_vector))


#ver cantidad de filas
map(df_listas, ~ length(.x[[1]]))


longitudes <- map_int(df_listas, ~ length(.x[[1]]))
min_len <- min(longitudes)


df_recortado <- df_listas %>%
  mutate(across(everything(), ~ lapply(.x, function(vec) vec[1:min_len]))) %>%
  unnest(cols = everything())


write.csv(df_recortado, 'df_1_final.csv', row.names = FALSE)

----------------------------------------------------------------------


for (i in seq_along(lista_dfs)) {
  write.csv(lista_dfs[[i]], paste0("df_", i, ".csv"), row.names = FALSE)
}




df_expandido <- df.1 %>%
  mutate(across(everything(), convertir_vector)) %>%
  unnest(cols = everything())
                
--------------------------------------------------------------------------------                

#historiales con fisica 


tabla.ff<- read_csv('historiales_con_Fisica_f.csv')

lista_dfs_ff <- split(tabla.ff, seq(nrow(tabla.ff)))

dff_1_f <- lista_dfs_ff[[1]]
dff_2 <- lista_dfs_f[[2]]
dff_3 <- lista_dfs_f[[3]]
dff_4 <- lista_dfs_f[[4]]
dff_5_f <- lista_dfs_ff[[5]]


convertir_vector <- function(x) {
  lapply(x, function(s) as.numeric(strsplit(gsub("\\[|\\]", "", s), ",")[[1]]))
}


-----------------------------------------------------------------------------
df_listas_1_f <- dff_1_f %>% 
  mutate(across(everything(), convertir_vector))

map(df_listas_1_f, ~ length(.x[[1]]))


longitudes_1 <- map_int(df_listas_1, ~ length(.x[[1]]))
longitud_final_1 <- longitudes['val_f']


df_listas_1_padded_f <- df_listas_1_f %>%
  mutate(across(everything(), ~ lapply(.x, function(vec) {
    len <- length(vec)
    if (len < longitud_final_1) {
      c(vec, rep(NA, longitud_final_1 - len))
    } else if (len > longitud_final_1) {
      vec[1:longitud_final_1]
    } else {
      vec
    }
  })))


df_expandido_1_f <- df_listas_1_padded_f %>%
  unnest(cols = everything())


write.csv(df_expandido_1, 'df_f_1_f.csv')

--------------------------------------------------------------------------------

df_listas_5_f <- dff_5_f %>% 
  mutate(across(everything(), convertir_vector))

map(df_listas_5_f, ~ length(.x[[1]]))


longitudes_5_f <- map_int(df_listas_5, ~ length(.x[[1]]))
longitud_final_5_f <- longitudes_5_f['val_f']


df_listas_5_padded_f <- df_listas_5_f %>%
  mutate(across(everything(), ~ lapply(.x, function(vec) {
    len <- length(vec)
    if (len < longitud_final_5_f) {
      c(vec, rep(NA, longitud_final_5_f - len))
    } else if (len > longitud_final_5_f) {
      vec[1:longitud_final_5_f]
    } else {
      vec
    }
  })))


df_expandido_5_f <- df_listas_5_padded_f %>%
  unnest(cols = everything())


write.csv(df_expandido_1, 'df_f_5_f.csv')





----------------------------------------------------------------------------
  
for (i in seq_along(lista_dfs_f)) {
  write.csv(lista_dfs_f[[i]], paste0("df_f_", i, ".csv"), row.names = FALSE)
}


