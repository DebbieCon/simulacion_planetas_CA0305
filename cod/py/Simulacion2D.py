import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Simulacion_N_Cuerpos import Simulacion_N_Cuerpos as snc
import numpy as np
import math 

class Simulacion2D(snc):

    def __init__(self, cuerpos: list, h, G = 6.67e-11):
        '''
        Instancia un objeto tipo Simulacion2D capaz de animar en dos dimensiones
        las trayectorias de los cuerpos astronómicos que recibe.

        Parámetros
        ----------
            cuerpos : list
                Lista de cuerpos instanciados por Cuerpo.py con todos sus parámetros iniciales, puede ser 1 o más
            h : int
                Tiempo que tarda la simulación (pasado este tiempo se repite)
            G : float
                Constante de Gravitación Universal, por defecto se usa 6.672 x 10-11 Nm2/kg2
        Retorna
        -------

        '''
        super().__init__(cuerpos, G, h)