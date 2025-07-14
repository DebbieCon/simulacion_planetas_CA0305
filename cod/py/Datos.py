import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class Datos(Dataset):
    def __init__(self, url, fecha='fecha', planetas=None):
      '''
      Constructor de la clase Datos, que hereda de Dataset
      
      Parámetros
      ----------
      url: str
        dirección a la base de datos 
      fecha: pd.datetime  
        fechas convertidas a objetos datetime 
      planetas: list 
        lista de planetas en la base 
      '''
      self._datos = pd.read_csv(url, index_col=0)

      if planetas is None:
        planetas = ['mercurio','venus','tierra','marte','jupiter','saturno','urano','neptuno']

      self._planetas = planetas
      self._num_planetas = len(planetas)
      self._num_datos = len(self._datos)

      self._t = None
      self._id_obj = None
      self._r = None
      self._v = None

      self._scaler_t = MinMaxScaler()
      self._scaler_pos = StandardScaler()
      self._scaler_r = StandardScaler()
      self._scaler_v = StandardScaler()

      self._t_s = None
      self._pos_s = None
      self._r_s = None
      self._v_s = None

      self._pos_mtx = None
      self._vel_mtx = None

      self._datos[fecha] = pd.to_datetime(self._datos[fecha])
      self._datos['t'] = (self._datos[fecha] - self._datos[fecha].min()).dt.days.astype(np.float32)
      self._datos.drop(columns=[fecha], inplace=True)
      self._t = self._datos['t'].values.reshape(-1,1).astype(np.float32)

    # --- Properties con setters ---
    @property
    
    def datos(self): 
      '''
      Retorna el DataFrame con la base de datos
     
      Retorna
      -------
      self.__datos: pandas.DataFrame 
        DataFrame con la base de datos 
      '''
    
      return self._datos
    
    @datos.setter
    
    def datos(self, valor): 
      '''
      Asigna un nuevo DataFrame al conjunto de datos originales
  
      Parámetros
      ----------
      valor : pandas.DataFrame
          Nuevo DataFrame que reemplaza al anterior
      '''
      self._datos = valor

    @property
    def planetas(self): 
      '''
      Retorna la lista de nombres de los planetas considerados
  
      Retorna
      -------
      self._planetas : list
          Lista con los nombres de los planetas
      '''
      return self._planetas
    
    @planetas.setter
    def planetas(self, valor): 
      '''
      Asigna una nueva lista de planetas
  
      Parámetros
      ----------
      valor : list
          Lista con los nuevos nombres de los planetas
      '''
      self._planetas = valor

    @property
    def num_planetas(self): 
      '''
      Retorna el número de planetas considerados 
  
      Retorna
      -------
      self._num_planetas : int
          Cantidad total de planetas considerados.
      '''
      return self._num_planetas
    @num_planetas.setter
    def num_planetas(self, valor): 
      '''
      Asigna un nuevo número de planetas 
  
      Parámetros
      ----------
      valor : int
          nuevo número total de planetas
      '''
      self._num_planetas = valor

    @property
    def num_datos(self): 
      '''
      Retorna el número de datos en el DataFrame.
  
      Retorna
      -------
      self._num_datos : int
          Cantidad total de datos
      '''
      return self._num_datos
    @num_datos.setter
    def num_datos(self, valor): 
      '''
      Asigna un nuevo número total de datos 
  
      Parámetros
      ----------
      valor : int
          nuevo número de registros en el conjunto de datos
      '''
      self._num_datos = valor

    @property
    def t(self): 
      '''
      Retorna los valores de tiempo 
  
      Retorna
      -------
      self._t : numpy.ndarray
          array de los tiempos en días desde la fecha inicial
      '''
      return self._t
    @t.setter
    def t(self, valor): 
      '''
      Asigna nuevos valores de tiempo 
  
      Retorna
      -------
      self._t : numpy.ndarray
          nuevo array de tiempos 
      '''
      self._t = valor

    @property
    def id_obj(self): 
      '''
       Retorna el identificador del objeto.
  
       Retorna
       -------
       self._id_obj: int 
          identificador númerico asociado al objeto 
       '''
      return self._id_obj

    @id_obj.setter
    def id_obj(self, valor): 
      '''
      Asigna un nuevo identificador al objeto.
  
      Parámetros
      ----------
      valor : int 
          Nuevo identificador del objeto.
      ''' 
      self._id_obj = valor

    @property
    def r(self): 
      '''
      Retorna el vector de posición del objeto
  
      Retorna
      -------
      self._r : np.array 
          Vector de posiciones 
      '''
      return self._r
    
    @r.setter
    def r(self, valor): 
      '''
      Asigna un nuevo vector de posición 
  
      Parámetros
      ----------
      valor : np.array 
          nuevo vector de posiciones 
      '''
      self._r = valor

    @property
    def v(self): 
      '''
      Retorna un vector con las velocidades
  
      Retorna
      -------
      self._v : np.array 
          Vector con las velocidades
      '''
      return self._v
    @v.setter
    def v(self, valor): 
      '''
      Asigna un nuevo vector de velocidades 
  
      Parámetros
      ----------
      valor : np.array 
          nuevo vector con velocidades
      '''
      self._v = valor

    @property
    def scaler_t(self): 
      '''
      Retorna el normalizador del tiempo 
  
      Retorna
      -------
      self._scaler_t : sklearn.preprocessing.MinMaxScaler
          normalizador de tiempo 
      '''
      return self._scaler_t
    @scaler_t.setter
    def scaler_t(self, valor): 
      '''
      Asigna un nuevo escalador de tiempo 
  
      Parámetros
      ----------
      valor : sklearn.preprocessing.MinMaxScaler
          nuevo normalizador del tiempo  
      '''
      self._scaler_t = valor

    @property
    def scaler_pos(self): 
      '''
      Retorna el escalador de las posiciones 
  
      Retorna
      -------
      self._scaler_pos : sklearn.preprocessing.StandardScaler
          nuevo normalizador de posiciones
      '''
      return self._scaler_pos
    @scaler_pos.setter
    def scaler_pos(self, valor): 
      '''
      Asigna un nuevo escalador para posiciones
  
      Parámetros
      ----------
      valor : sklearn.preprocessing.StandardScaler
          nuevo escalador para las posiciones 
      '''
      self._scaler_pos = valor

    @property
    def scaler_r(self): 
      '''
      Retorna el escalador de las posiciones del objeto objetivo 
  
      Retorna
      -------
      self._scaler_r : sklearn.preprocessing.StandardScaler
          esclador de posiciones 
      '''
      return self._scaler_r
    @scaler_r.setter
    def scaler_r(self, valor): 
      '''
      Asigna nuevo escalador de posiciones 
  
      Parámetros
      ----------
      valor : sklearn.preprocessing.StandardScaler
          Nuevo escalador de posiciones 
      '''
      self._scaler_r = valor

    @property
    def scaler_v(self): 
      '''
      Retorna el normalizador de velocidades
  
      Retorna
      -------
      self._scaler_v : sklearn.preprocessing.StandardScaler
          normalizador de los valores de velocidad 
      '''
      return self._scaler_v
    @scaler_v.setter
    def scaler_v(self, valor): 
      '''
      Asigna un nuevo escalador para las velocidades  
  
      Parámetros
      ----------
      valor : sklearn.preprocessing.StandardScaler
          Escalador para las velocidades.
      '''
      self._scaler_v = valor

    @property
    def t_s(self): 
      '''
      Retorna los valores de tiempo escalados
  
      Retorna
      -------
      self._t_s : numpy.ndarray
          tiempos escalados con el MinMaxScaler
      '''
      return self._t_s
    @t_s.setter
    def t_s(self, valor): 
      '''
      Asigna nuevos valores de tiempo escalado
  
      Parámetros
      ----------
      valor : numpy.ndarray
          nuevos valores de tiempo ya normalizados
      '''
      self._t_s = valor

    @property
    def pos_s(self): 
      '''
      Retorna las posiciones escaladas 
  
      Retorna
      -------
      self._pos_s : numpy.ndarray
          Matriz de posiciones normalizadas
      '''
      return self._pos_s
    @pos_s.setter
    def pos_s(self, valor): 
      '''
      Asigna nuevas posiciones escaladas
  
      Parámetros
      ----------
      valor : numpy.ndarray
          nueva matriz de posiciones normalizadas.
      '''
      self._pos_s = valor

    @property
    def r_s(self): 
      '''
      Retorna las posiciones escaladas
  
      Retorna
      -------
      self._r_s : numpy.ndarray
          posiciiones normalizadas 
      '''
      return self._r_s
    @r_s.setter
    def r_s(self, valor):
      '''
      Asigna nuevas posiciones escaladas.
  
      Parámetros
      ----------
      valor : numpy.ndarray
          nuevos valores de las posiciones normalizadas 
      '''
      self._r_s = valor

    @property
    def v_s(self): 
      '''
      Retorna las velocidades escaladas
  
      Retorna
      -------
      self._v_s : numpy.ndarray
          Vector con las velocidades normalizadas
      '''
      return self._v_s
    @v_s.setter
    def v_s(self, valor):
      '''
      Asigna nuevas velocidades escaladas
  
      Parámetros
      ----------
      valor : numpy.ndarray
          nuevo vector de velocidades ya normalizadas
      '''
      self._v_s = valor

    @property
    def pos_mtx(self):
      '''
      Retorna la matriz con las posiciones originales.
  
      Retorna
      -------
      self._pos_mtx : numpy.ndarray
          Matriz con posiciones de los planetas 
      '''
      return self._pos_mtx
    @pos_mtx.setter
    def pos_mtx(self, valor):
      '''
      Asigna una nueva matriz de posiciones
  
      Parámetros
      ----------
      valor : numpy.ndarray
          Matriz con las posiciones de los objetos.
      '''
      self._pos_mtx = valor

    @property
    def vel_mtx(self): 
      '''
      Retorna la matriz con las velocidades originales.
  
      Retorna
      -------
      self._vel_mtx : numpy.ndarray
          Matriz con las velocidades de los planetas 
      '''
      return self._vel_mtx
    @vel_mtx.setter
    def vel_mtx(self, valor): 
      '''
      Asigna una nueva matriz de velocidades
  
      Parámetros
      ----------
      valor : numpy.ndarray
          Matriz con las velocidades de los planetas 
      '''
      self._vel_mtx = valor

    # --- Métodos de procesamiento ---
    def posiciones_velocidades(self, obj='tierra'):
      '''
      Calcula y organiza las posiciones y velocidades de todos los planetas
  
      Parámetros
      ----------
      obj : str
          Nombre del planeta objetivo para el análisis individual ('tierra').
  
      Retorna 
      -------
      self.r : numpy.ndarray
          Posiciones del planeta objetivo en cada tiempo
      self.v : numpy.ndarray
          Velocidades del planeta objetivo en cada tiempo
      self.pos_mtx : numpy.ndarray
          Matriz con posiciones de todos los planetas
      self.vel_mtx : numpy.ndarray
          Matriz con velocidades de todos los planetas
      self.id_obj : int
          Índice del planeta objetivo 
      '''
      pos = np.zeros((self.num_datos, self.num_planetas, 3), dtype=np.float32)
      vel = np.zeros((self.num_datos, self.num_planetas, 3), dtype=np.float32)
      
      for i, p in enumerate(self.planetas):
        pos[:, i, 0] = self.datos[f'{p}_x']
        pos[:, i, 1] = self.datos[f'{p}_y']
        pos[:, i, 2] = self.datos[f'{p}_z']
        vel[:, i, 0] = self.datos[f'{p}_vx']
        vel[:, i, 1] = self.datos[f'{p}_vy']
        vel[:, i, 2] = self.datos[f'{p}_vz']
        
      self.id_obj = self.planetas.index(obj)
      self.r = pos[:, self.id_obj, :]
      self.v = vel[:, self.id_obj, :]
      self.pos_mtx = pos
      self.vel_mtx = vel
      
      return self.id_obj, self.r, self.v, self.pos_mtx, self.vel_mtx 



    def escalar_y_transformar(self):
      '''
      Normaliza los valores de la velocidades, tiempo, y posición
      
      Retorna
      -------
      self.t_s : torch.Tensor
          Tiempos escalados.
      self.pos_s : torch.Tensor
          Posiciones escaladas de todos los planetas
      self.r_s : torch.Tensor
          Posiciones escaladas del planeta objetivo
      self.v_s : torch.Tensor
          Velocidades escaladas del planeta objetivo
      '''
      self.scaler_t.fit(self.t)
      self.scaler_pos.fit(self.pos_mtx.reshape(self.num_datos, -1))
      self.scaler_r.fit(self.r)
      self.scaler_v.fit(self.v)
      
      self.t_s = torch.tensor(self.scaler_t.transform(self.t), dtype=torch.float32)
      self.pos_s = torch.tensor(
          self.scaler_pos.transform(self.pos_mtx.reshape(self.num_datos, -1)).reshape(self.num_datos, self.num_planetas, 3),
          dtype=torch.float32
          )
          
      self.r_s = torch.tensor(self.scaler_r.transform(self.r), dtype=torch.float32)
      self.v_s = torch.tensor(self.scaler_v.transform(self.v), dtype=torch.float32)
      
      return self.t_s, self.pos_s, self.r_s, self.v_s



    def inv_escalar_pos(self, r_escalado):
      '''
      Invierte la normaizacion de las posiciones del planeta objetivo
  
      Parámetros
      ----------
      r_escalado : torch.Tensor
          Posiciones escaladas a transformar a valores originales
  
      Retorna
      -------
      r_original : numpy.ndarray
          Posiciones sin escalar del planeta objetivo
      '''
      r_np = r_escalado.detach().cpu().numpy()
      return self.scaler_r.inverse_transform(r_np)



    def masas_planetas_dic(self):
      '''
      Retorna la lista de masas de los planetas
  
      Retorna
      -------
      masas : list (float)
          Lista de masas (en kilogramos) de los planetas
      '''
      masas_dict = {
          "sol": 1.9885e30, "mercurio": 3.301e23, "venus": 4.867e24,
          "tierra": 5.972e24, "marte": 6.417e23, "jupiter": 1.898e27,
          "saturno": 5.683e26, "urano": 8.681e25, "neptuno": 1.024e26
      }
      return [masas_dict[p.lower()] for p in self.planetas]



    def datos_id(self, idx):
      '''
      Retorna un conjunto de datos escalado a partir de un índice
  
      Parámetros
      ----------
      idx : int
          indice 
  
      Retorna
      -------
      entrada : tuple
          Tupla con (tiempo escalado, posiciones escaladas de todos los planetas)
      salida : tuple
          Tupla con (posición escalada del planeta objetivo, velocidad escalada del planeta objetivo)
      '''
      return (self.t_s[idx], self.pos_s[idx]), (self.r_s[idx], self.v_s[idx])
  
  
    
    def __len__(self):
      '''
      Retorna la cantidad total de datos 
  
      Retorna
      -------
      self.num_datos : int
          número total de observaciones 
      '''
      return self.num_datos



    def __getitem__(self, idx):
      '''
      Permite acceder a un elemento específico del dataset 
  
      Parámetros
      ----------
      idx : int
          Índice del elemento
  
      Retorna
      -------
      entrada : tuple
          Tupla con (tiempo escalado, posiciones escaladas de todos los planetas)
      salida : tuple
          Tupla con (posición escalada del planeta objetivo, velocidad escalada del planeta objetivo)
      '''
      return (self.t_s[idx], self.pos_s[idx]), (self.r_s[idx], self.v_s[idx])
