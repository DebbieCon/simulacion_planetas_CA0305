import numpy as np
from Cuerpo import Cuerpo

class Simulacion_N_Cuerpos:
    
    def __init__(self, cuerpos, G, h):
        '''
        Inicializa un objeto tipo simulador que emplea el método de Runge Kutta de orden 4.
        
        Parámetros
        ----------
            cuerpos : list
                Lista de cuerpos con sus características, generados con la clase Cuerpo.
            G : float
                Valor de la constante de Gravitación Universal, por defecto se usa 6.67430e-11 Nm2/kg2
            h : float
                Tiempo de paso, necesario para el método de Runge Kutta
        
        Retorna
        -------
            None
        '''
        self._cuerpos = cuerpos
        self._G = 6.67430e-11 
        self._h = h 
        self._trayectorias = [ [] for i in cuerpos]

    def __str__(self):
        '''
        Retorna una representación en texto del estado actual de la simulación.

        Retorna
        -------
            str
                Información de los cuerpos y parámetros principales.
        '''
        info = f"Simulación N-Cuerpos\nConstante G: {self._G}\nPaso de tiempo h: {self._h}\n"
        info += f"Número de cuerpos: {len(self._cuerpos)}\n"
        for i, cuerpo in enumerate(self._cuerpos):
            info += f"Cuerpo {i}: Posición={cuerpo.pos}, Velocidad={cuerpo.vel}, Masa={cuerpo.masa}\n"
        return info

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
        Retorna
        -------
        '''
        self._cuerpos = nuevos_cuerpos
        self._trayectorias = [[] for _ in nuevos_cuerpos]

    @property
    def G(self):
        '''
        Retorna la constante de gravitación universal.

        Parámetros
        ----------

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
        
        Retorna
        -------
        '''
        self._G = nuevo_G

    @property
    def h(self):
        '''
        Retorna el tiempo de paso para el método de Runge Kutta.

        Parámetros
        ----------

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
        
        Retorna
        -------
        '''
        self._h = nuevo_h

    @property
    def trayectorias(self):
        '''
        Retorna las trayectorias de los cuerpos calculadas por el simulador.

        Parámetros
        ----------

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
        
        Retorna
        -------
        '''
        self._trayectorias = nuevas_trayectorias
    
    
    def actualizar_posiciones_temp(self, nuevas_posiciones):
        '''
        Actualiza temporalmente las posiciones de los cuerpos para los cálculos intermedios de Runge-Kutta.

        Parámetros
        ----------
            nuevas_posiciones : np.ndarray
                Arreglo con las nuevas posiciones temporales para cada cuerpo.
        
        Retorna
        -------
        '''
        for i, cuerpo in enumerate(self._cuerpos):
            cuerpo._pos = nuevas_posiciones[i]

    
    def calcular_aceleraciones(self):
        '''
        Calcula la aceleración en los tres ejes (x, y, z) para cada cuerpo, considerando la influencia gravitacional de los demás cuerpos.

        Parámetros
        ----------
        
        Retorna
        -------
            aceleraciones : np.ndarray
                Arreglo con la aceleración en los 3 ejes para cada cuerpo.
        '''
        num_cuerpos = len(self._cuerpos)
        aceleraciones = np.zeros((num_cuerpos, len(self._cuerpos[0].pos))) # Permite ya sea de dos o tres dimensiones
        for i in range(num_cuerpos):
            for j in range(num_cuerpos):
                if i != j:
                    r_ij = self._cuerpos[j].pos - self._cuerpos[i].pos # Vector r_ij
                    dist_ij = np.linalg.norm(r_ij) + 1e-10 # Norma euclídea
                    aceleraciones[i] += self._G * self._cuerpos[j].masa * r_ij / (dist_ij**3)
        
        for i, cuerpo in enumerate(self._cuerpos):
            cuerpo._acel = aceleraciones[i]
            
        return aceleraciones
   
    # Métodos necesarios para la aproximación de la ODE por medio de Runge Kutta 4

    def paso_rk4(self):
        '''
        Realiza un paso de integración utilizando el método de Runge-Kutta de orden 4 (RK4) para actualizar las posiciones y velocidades de los cuerpos.

        Parámetros
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
        Ejecuta la simulación durante un número dado de pasos, actualizando las posiciones y velocidades de los cuerpos en cada paso.

        Parámetros
        ----------
            pasos : int
                Número de pasos de integración a realizar.

        Retorna
        -------
        '''
        for paso in range(pasos):
            self.paso_rk4()





