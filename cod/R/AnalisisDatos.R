library(tidyverse)
library(lubridate)
library(dplyr)
library(kableExtra)

df <- read.csv("data\\resultados.csv")

source("cod\\R\\LimpiezaDatos.R")

fecha.inicial <- ymd("2025-04-01") # Cambiar según el caso
df$t..días. <- fecha.inicial + days(0:(nrow(df)-1))
names(df)[names(df) == "t..días."] <- "fecha"
names(datos.reales)[names(datos.reales) == "Calendar.Date..TDB."] <- "fecha"


df_analisis <- left_join(df, datos.reales, by = "fecha")

# Hallar la norma de la velocidad y la posicion para ambas tablas

df_analisis <- df_analisis %>% 
  mutate(
    r_calc = sqrt((r_x^2 + r_y^2 + r_z^2)),
    r_real = sqrt((X^2 + Y^2 + Z^2)),
    v_calc = sqrt((v_x^2 + v_y^2 + v_z^2)),
    v_real = sqrt((VX^2 + VY^2 + VZ^2))
  )

df_analisis <- df_analisis %>%
    select(fecha, r_calc, r_real, v_calc, v_real)

df_analisis <- df_analisis %>%
    mutate(
        error_r = (abs(r_calc - r_real) / r_real) * 100,
        error_v = (abs(v_real - v_calc) / v_real) * 100
    )

df_analisis <- df_analisis %>%
    select(fecha, r_real, r_calc, error_r, v_real, v_calc, error_v)    



names(df_analisis) <- c("fecha", "pos_real", "pos_cal", "error_pos", "vel_real", "vel_calc", "error_vel")
# Guardar el dataframe final en un archivo CSV
 write.csv(df_analisis, "data\\df_analisis.csv", row.names = FALSE)
# Mostrar el dataframe final




