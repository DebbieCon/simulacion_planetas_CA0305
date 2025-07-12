import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from Simulacion_N_Cuerpos import Simulacion_N_Cuerpos as snc
import numpy as np

class Simulacion3D(snc):

    def __init__(self, cuerpos, h, G = 6.67e-11):
        '''
        Instancia un objeto tipo Simulacion3D capaz de animar en tres dimensiones
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
        Retorna una representación en cadena de la simulación 3D.

        Parámetros
        ----------
        
        Retorna
        -------
            str
                Representación en cadena de la simulación 3D.
        '''
        return f"Simulación 3D con {len(self._cuerpos)} cuerpos, paso de tiempo: {self._h} s, G: {self._G} m^3 kg^-1 s^-2"

    @property
    def animacion(self):
        '''
        Retorna la animación de la simulación 3D.

        Parámetros
        ----------

        Retorna
        -------
            self._animacion : FuncAnimation
                Animación de la simulación 3D, se usa para mostrar la animación en pantalla.
        '''
        return self._animacion
    
    @animacion.setter
    def animacion(self, value):
        '''
        Establece la animación de la simulación 3D.

        Parámetros
        ----------
            value : FuncAnimation
                Animación de la simulación 3D.
        Retorna
        -------
        '''
        self._animacion = value


    def animar(self):
        '''
        Crea una animación 3D de las trayectorias de los cuerpos en la simulación.
        Utiliza matplotlib.animation.FuncAnimation para crear la animación de las trayectorias de los cuerpos.
        
        Parámetros
        ----------

        Retorna
        -------
        '''

        fig = plt.figure(figsize = (10,8))
        ax = fig.add_subplot(111,projection='3d')
        
        trayectorias = self._trayectorias
        n_cuerpos = len(trayectorias)

        # Configuración 3D
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'Simulación 3D de {n_cuerpos}-Cuerpos')
        
        
        # Calcular límites automáticamente
        all_coords = np.array([p for cuerpo in trayectorias for p in cuerpo])
        max_range = np.max(np.abs(all_coords)) * 1.1  # Margen del 10%
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)
        
        ax.set_box_aspect([1, 1, 1])
        
        
        # Líneas y puntos
        lineas = [ax.plot([], [], [], '-', label=f'Cuerpo {i}')[0] for i in range(n_cuerpos)]
        puntos = [ax.plot([], [], [], 'o')[0] for _ in range(n_cuerpos)]

        def unir():
            for linea, punto in zip(lineas, puntos):
                linea.set_data([], [])
                linea.set_3d_properties([])
                punto.set_data([], [])
                punto.set_3d_properties([])
            return lineas + puntos

        def actualizar(frame):
            for i, (linea, punto) in enumerate(zip(lineas, puntos)):
                coords = np.array(trayectorias[i][:frame])
                if len(coords) > 0:
                    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
                    linea.set_data(x, y)
                    linea.set_3d_properties(z)
                    punto.set_data([x[-1]], [y[-1]])
                    punto.set_3d_properties([z[-1]])
            return lineas + puntos

        anim = animation.FuncAnimation(
            fig,
            actualizar,
            frames=min(len(trayectoria) for trayectoria in trayectorias),
            init_func=unir,
            blit=False,
            interval=20
        )


        plt.legend()
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
        self.animacion.save(nombre_archivo, writer='ffmpeg', fps=30)
        print(f"Animación guardada como {nombre_archivo}")