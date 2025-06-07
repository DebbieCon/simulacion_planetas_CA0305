import numpy as np
from Cuerpo import Cuerpo

class Simulacion_N_Cuerpos:
    
    def __init__(self, cuerpos, G, h):
        '''
        Inicializa un objeto tipo simualdor quien empleara el metodo de Runge Kutta de orden 4.
        
        Parámetros
        ----------
        
            cuerpos : list
                Lista de cuerpos con sus caracteristicas, generados con la clase Cuerpos.py
            G : float
                Valor de la constante de Gravitación Universal, por defecto se usa 6.67430e-11 Nm2/kg2
            h : float
                Tiempo de paso, necesario para el método de Runge Kutta
        
        Retorna
        -------
        
        '''
        self._cuerpos = cuerpos
        self._G = 6.67430e-11 
        self._h = h 
        self._trayectorias = [ [] for i in cuerpos]


    @property
    def cuerpos(self):
        '''
        Retorna la lista de cuerpos del simulador.

        Retorna
        -------
            list
                Lista de objetos tipo Cuerpo.
        '''
        return self._cuerpos

    @cuerpos.setter
    def cuerpos(self, nuevos_cuerpos):
        '''
        Asigna una nueva lista de cuerpos al simulador.

        Parámetros
        ----------
            nuevos_cuerpos : list
                Nueva lista de cuerpos.
        '''
        self._cuerpos = nuevos_cuerpos
        self._trayectorias = [[] for _ in nuevos_cuerpos]

    @property
    def G(self):
        '''
        Retorna la constante de gravitación universal.

        Retorna
        -------
            float
                Valor de la constante G en Nm²/kg².
        '''
        return self._G

    @G.setter
    def G(self, nuevo_G):
        '''
        Asigna un nuevo valor a la constante de gravitación universal.

        Parámetros
        ----------
            nuevo_G : float
                Nuevo valor para la constante G.
        '''
        self._G = nuevo_G

    @property
    def h(self):
        '''
        Retorna el tiempo de paso para el método de Runge Kutta.

        Retorna
        -------
            float
                Valor del tiempo de paso.
        '''
        return self._h

    @h.setter
    def h(self, nuevo_h):
        '''
        Asigna un nuevo valor al tiempo de paso.

        Parámetros
        ----------
            nuevo_h : float
                Nuevo tiempo de paso.
        '''
        self._h = nuevo_h

    @property
    def trayectorias(self):
        '''
        Retorna las trayectorias de los cuerpos calculadas por el simulador.

        Retorna
        -------
            list
                Lista de listas que representan la trayectoria de cada cuerpo.
        '''
        return self._trayectorias

    @trayectorias.setter
    def trayectorias(self, nuevas_trayectorias):
        '''
        Asigna nuevas trayectorias a los cuerpos del simulador.

        Parámetros
        ----------
            nuevas_trayectorias : list
                Lista de trayectorias para los cuerpos.
        '''
        self._trayectorias = nuevas_trayectorias
    
    
    
    def actualizar_posiciones_temp(self, nuevas_posiciones):
        '''
        Parametros
        ----------
            nuevas_posiciones: list 
              lista con las nuevas posiciones temporales asignadas 
        
        Retorna
        -------
        
        '''
        for i, cuerpo in enumerate(self._cuerpos):
            cuerpo._pos = nuevas_posiciones[i]

    
    def calcular_aceleraciones(self):
        '''
        Calcula la aceleracion en los tres ejes (x,y,z) para el cuerpo i, note que la aceleracion que siente i es debido a los n-1
        cuerpos restantes
        
        Parametros
        ----------
        
        Retorna
        --------
            aceleraciones : lista
                Lista con la aceleracion en los 3 ejes
        '''
        num_cuerpos = len(self._cuerpos)
        aceleraciones = np.zeros((num_cuerpos, len(self._cuerpos[0].pos))) #Permite ya sea de dos o tres dimensiones
        for i in range(num_cuerpos):
            for j in range(num_cuerpos):
                if i != j:
                    r_ij= self._cuerpos[j].pos - self._cuerpos[i].pos #Aca genero el vector r_ij
                    #Se le debe hallar la norma euclidea
                    dist_ij = np.linalg.norm(r_ij) + 1e-10
                    aceleraciones[i] += self._G * self._cuerpos[j].masa*r_ij / (dist_ij**3)
        
        for i, cuerpo in enumerate(self._cuerpos):
            cuerpo._acel = aceleraciones[i]
            
        return aceleraciones
   
    #Se haran los metodos necesarios para la aproximacion de la ODE por medio de Runge Kutta 4

    def paso_rk4(self):
      ''' 
      Se calcula la nueva posicion y velocidad de los cuerpos usando el metodo de Runge Kutta 4
      
      Parametros 
      ----------
      
      Retorna
      -------
  
      '''
      
        pos_original = np.array([c.pos for c in self._cuerpos])
        vel_original = np.array([c.vel for c in self._cuerpos])

        # k1
        k1_v = self.calcular_aceleraciones()
        k1_r = vel_original

        # k2
        pos_temp = pos_original + 0.5 * self._h * k1_r
        self.actualizar_posiciones_temp(pos_temp)
        k2_v = self.calcular_aceleraciones()
        k2_r = vel_original + 0.5 * self._h * k1_v

        # k3
        pos_temp = pos_original + 0.5 * self._h * k2_r
        self.actualizar_posiciones_temp(pos_temp)
        k3_v = self.calcular_aceleraciones()
        k3_r = vel_original + 0.5 * self._h * k2_v

        # k4
        pos_temp = pos_original + self._h * k3_r
        self.actualizar_posiciones_temp(pos_temp)
        k4_v = self.calcular_aceleraciones()
        k4_r = vel_original + self._h * k3_v

        # Restaurar posiciones originales
        self.actualizar_posiciones_temp(pos_original)

        # Actualizar posiciones y velocidades finales
        nuevas_pos = pos_original + (self._h/6) * (k1_r + 2*k2_r + 2*k3_r + k4_r)
        nuevas_vel = vel_original + (self._h/6) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        for i, cuerpo in enumerate(self._cuerpos):
            cuerpo._pos = nuevas_pos[i]
            cuerpo._vel = nuevas_vel[i]
            self._trayectorias[i].append(nuevas_pos[i].copy())
            
    def simular(self, pasos=1000):
      ''' 
      Intancia el metodo paso_rk4
      
      Parametros 
      ---------
         pasos : int
            numero de pasos temporales, esta fijo en 1000
      
      Retorna
      -------
      
      '''
        for paso in range(pasos):
            self.paso_rk4()


        
