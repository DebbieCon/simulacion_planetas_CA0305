library(tidyverse)
library(lubridate)
library(dplyr)

# Función para cargar el txt

cargar_csv <- function(ruta){
  # Cargar el csv
  base <- read.csv(ruta, header = TRUE)
  # Eliminar la primera columna
  base <- base %>% select(-JDTDB) %>%  select(-X.1)
  base <- base[1:9133,]
}

formato_col <- function(base, nombre){
  # Dar formato a las fechas
  
  fechas_limpias <- gsub("^A\\.D\\.\\s+","", base$Fecha)
  base$Fecha <- parse_date_time(
    fechas_limpias, orders = "Y-b-d HMS", locale = "en_us.utf8"
  )
  base$Fecha <- format(base$Fecha, "%Y-%m-%d")
  
  # Cambiar nombre de las columnas
  colnames(base) <- c("fecha", paste0(nombre,"_x"),
   paste0(nombre,"_y"),paste0(nombre,"_z"),paste0(nombre,"_vx"),
    paste0(nombre,"_vy"),paste0(nombre,"_vz"))
  return(base)
}

unir_bases <- function(bases, columna){
  
  inicial <- bases[[1]]
  
  for (base in bases[-1]){
    inicial <- left_join(inicial, base, by = columna)
  }
  return(inicial)
}

# Mercurio
merc <- cargar_csv("data/mercurio_jp.csv")
merc <- formato_col(merc, "mercurio")

# Venus
ven <- cargar_csv("data/venus_jp.csv")
ven <- formato_col(ven, "venus")

# Tierra
tie <- cargar_csv("data/tierra_jp.csv")
tie <- formato_col(tie, "tierra")

# Marte
mar <- cargar_csv("data/marte_jp.csv")
mar <- formato_col(mar, "marte")

# Júpiter
jup <- cargar_csv("data/jupiter_jp.csv")
jup <- formato_col(jup, "jupiter")

# Saturno
sat <- cargar_csv("data/saturno_jp.csv")
sat <- formato_col(sat, "saturno")

# Urano
ura <- cargar_csv("data/urano_jp.csv")
ura <- formato_col(ura, "urano")

# Neptuno
nep <- cargar_csv("data/neptuno_jp.csv")
nep <- formato_col(nep, "neptuno")

planetas <- list(merc, ven, tie, mar, jup, sat, ura, nep)

planetas <- unir_bases(planetas, "fecha")

write.csv(planetas, "data/sistema_solar.csv")
