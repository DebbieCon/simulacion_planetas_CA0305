import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from BaseDatos import BaseDatos


class PINN(keras.Model, BaseDatos):

    #Constructor
    def __init__(self, url, capas : list, activation : str, lrn_rate : float, peso_fis : float , masas : dict):

        super().__init__(url)

        self.__capas = capas
        self.__activacion = activation
        self.__lrn_rate = lrn_rate
        self.__peso_fis = peso_fis
        self.__G = 6.67430e-11
        self.__masas = masas
        self.__model = None
        self.__optimizador = None

    def crear_modelo(self):

        self.__model = self._build
    