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
        self._animacion = None
    
    def __str__(self):
        '''
        Retorna una representación en cadena de la simulación 2D.

        Retorna
        -------
            str
                Representación en cadena de la simulación 2D.
        '''
        return f"Simulación 2D con {len(self._cuerpos)} cuerpos, paso de tiempo: {self._h} s, G: {self._G} m^3 kg^-1 s^-2"

    @property
    def animacion(self):
        '''
        Retorna la animación de la simulación 2D.

        Parámetros
        ----------

        Retorna
        -------
            self._animacion : FuncAnimation
                Animación de la simulación 2D, se usa para mostrar la animación en pantalla.
        '''
        return self._animacion  

    @animacion.setter
    def animacion(self, new_animacion):
        '''
        Asigna una nueva animación a la simulación 2D.  

        Parámetros
        ----------
            new_animacion : FuncAnimation
                Nueva animación de la simulación 2D, se usa para mostrar la animación en pantalla.
        Retorna
        -------
        '''
        self._animacion = new_animacion
    
   
    def animar(self):
        '''
        Animación de la simulación en 2D, se usa matplotlib.animation.FuncAnimation
        para crear una animación de las trayectorias de los cuerpos.

        Parámetros
        ----------
            
        Retorna
        -------
            
        '''
        trayectorias = self._trayectorias
        n_cuerpos = len(trayectorias)
        
        fig, ax = plt.subplots(figsize=(10, 8))

        all_coords = np.array([p for cuerpo in trayectorias for p in cuerpo])
        max_range = np.max(np.abs(all_coords)) * 1.1  # Margen del 10%
        plt.xlim(-max_range, max_range)
        plt.ylim(-max_range, max_range)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Simulación 2D de N-Cuerpos')
        ax.grid(True)
        
        # Líneas (trayectorias) y puntos (cuerpos)
        lineas = [ax.plot([], [], '-', label=f'Cuerpo {i}')[0] 
                  for i in range(n_cuerpos)]
        puntos = [ax.plot([], [], 'o', markersize=8)[0] 
                  for i in range(n_cuerpos)]
        
        def unir():
            '''
            Inicializa los elementos graficos: lineas y puntos (cuerpos)
            
            Parametros
            ---------
            
            Retorna 
            -------
                list 
                lista que combina las lineas y los puntos actualizados 
            
            '''
            for linea, punto in zip(lineas, puntos):
                linea.set_data([], [])
                punto.set_data([], [])
            return lineas + puntos
        
        def actualizar(frame):
            '''
            Actualiza las lineas de trayectoria y los puntos de posicion 
            
            Parametros 
            ---------
                frame: int
                indice del marco de tiempo actual 
                
            Retorna
            -------
                list
                Lista que combina las lineas de trayectoria y los puntos de posicion actualizados 
            '''
            for i, (linea, punto) in enumerate(zip(lineas, puntos)):
                # Trayectoria hasta el frame actual
                x = [p[0] for p in trayectorias[i][:frame+1]]
                y = [p[1] for p in trayectorias[i][:frame+1]]
                linea.set_data(x, y)
                # Posición actual del cuerpo
                if frame < len(trayectorias[i]):
                    punto.set_data([trayectorias[i][frame][0]], 
                                   [trayectorias[i][frame][1]])
            return lineas + puntos
        
        anim = animation.FuncAnimation(
            fig, actualizar, frames=len(trayectorias[0]), 
            init_func=unir, blit=True, interval=20
        )
        plt.legend(loc='upper right')  
        plt.tight_layout()
        plt.show()

        self.animacion = anim  # Guardar la animación en el atributo


    def guardar_animacion(self, nombre_archivo : str):
        '''
        Guarda la animación en un archivo.

        Parámetros
        ----------
            nombre_archivo : str
                Nombre del archivo donde se guardará la animación.
        Retorna
        -------
            
        '''
        self._animacion.save(nombre_archivo, writer='ffmpeg', fps=15,bitrate=500)
        print(f"Animación guardada como {nombre_archivo}")

