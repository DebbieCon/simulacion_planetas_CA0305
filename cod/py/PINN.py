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
      '''
      Constructor de la clase PINN que hereda de la clase keras.Model y BaseDatos
      
      Parámetros
      ----------
      url : str
          URL del archivo de datos
      capas : list
          lista con el número de neuronas por capa
      activation : str
          nombre de la función de activación 
      lrn_rate : float
          tasa de aprendizaje para el optimizador
      peso_fis : float
          peso del término de la pérdida física en la función de pérdida total
      masas : dict
          diccionario con las masas de los cuerpos involucrados
      '''

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
      '''
      Crea y asigna el modelo de la red neuronal  
      '''
      self.__model = self._build
      
      
      
      
    
