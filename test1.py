import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Inicializar un gráfico vacío
fig, ax = plt.subplots()
line, = ax.plot([], [], label='Datos de Rendimiento')
ax.legend()

# Leer datos de rendimiento desde el archivo
with open('performance_data_Recon.txt', 'r') as performance_file:
    data = performance_file.readlines()

# Procesar los datos (supongo que son valores numéricos separados por comas)
x = [float(entry.split(',')[0]) for entry in data]
y = [float(entry.split(',')[1]) for entry in data]

# Función de inicialización para crear el gráfico vacío
def init():
    line.set_data([], [])
    return line,

# Función de actualización del gráfico
def update(frame):
    # Actualizar los datos en el gráfico utilizando el índice del frame
    line.set_data(x[:frame], y[:frame])
    
    return line,

# Obtener la cantidad de cuadros (frames)
frames = len(x)

# Crear animación con la cantidad de cuadros explícita
ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)

# Mostrar el gráfico
plt.show()
