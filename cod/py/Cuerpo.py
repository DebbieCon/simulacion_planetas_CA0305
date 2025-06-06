import numpy as np

class Cuerpo:

    def __init__(self, pos, vel, masa):
        '''
        Inicializa un objeto tipo cuerpo con una respectiva posición, velocidad y masa.

        Parámetros
        ----------
            pos : np.array
                vector de 3 coordenadas indicando la posición actual del cuerpo. 
            vel : dbl
                valor numérico de la velocidad medida en metros/segundos del cuerpo.
            masa : dbl
                valor numérico de la masa del cuerpo, medida en kilogramos.
        Retorna
        -------
        '''
        self._pos = np.array(pos, dtype = float)
        self._vel = np.array(vel, dtype = float)
        self._masa = masa
        self._acel = np.zeros_like(self._pos)
        
    @property
    def pos(self):
        '''
        Retorna la posición actual del cuerpo.

        Retorna
        -------
            np.array
                Vector de 3 coordenadas indicando la posición del cuerpo.
        '''
        return self._pos

    @pos.setter
    def pos(self, nueva_pos):
        '''
        Asigna una nueva posición al cuerpo.

        Parámetros
        ----------
            nueva_pos : array-like
                Vector de 3 coordenadas que representa la nueva posición.
        '''
        self._pos = np.array(nueva_pos, dtype=float)

    @property
    def vel(self):
        '''
        Retorna la velocidad actual del cuerpo.

        Retorna
        -------
            np.array
                Vector de 3 coordenadas que representa la velocidad del cuerpo en m/s.
        '''
        return self._vel

    @vel.setter
    def vel(self, nueva_vel):
        '''
        Asigna una nueva velocidad al cuerpo.

        Parámetros
        ----------
            nueva_vel : array-like
                Vector de 3 coordenadas que representa la nueva velocidad en m/s.
        '''
        self._vel = np.array(nueva_vel, dtype=float)

    @property
    def masa(self):
        '''
        Retorna la masa del cuerpo.

        Retorna
        -------
            float
                Masa del cuerpo en kilogramos.
        '''
        return self._masa

    @masa.setter
    def masa(self, nueva_masa):
        '''
        Asigna un nuevo valor a la masa del cuerpo.

        Parámetros
        ----------
            nueva_masa : float
                Nuevo valor de masa en kilogramos.
        '''
        self._masa = nueva_masa
