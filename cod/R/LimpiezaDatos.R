install.packages("tidyverse")
library(tidyverse)
install.packages("lubridate")
library(lubridate)
install.packages("dplyr")
library(dplyr)

# Cargar el csv
datos.reales <- read.csv("data\\datos_reales.txt")

# Escoger las columnas que nos interesan
## En este caso se escogen las columnas de fecha, posición y velocidad
datos.reales <- datos.reales %>% select(Calendar.Date..TDB., X, Y, Z, VX, VY, VZ)

# Colocar el formato de la fecha
datos.reales$Calendar.Date..TDB. <- gsub("A\\.D\\. ", "", datos.reales$Calendar.Date..TDB.)
datos.reales$Calendar.Date..TDB. <- as.Date(ymd_hms(datos.reales$Calendar.Date..TDB.))