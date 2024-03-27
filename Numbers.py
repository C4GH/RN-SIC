import time

# Define la ruta al archivo de entrada y la ruta al archivo de salida
ruta_archivo = 'C:\\Users\\afloresre\\Documents\\Cagh\\Red\\vectors400.txt'
ruta_archivo_salida = 'C:\\Users\\afloresre\\Documents\\Cagh\\Red\\numeros_iniciales.txt'

# Inicia el tiempo de ejecución
inicio = time.time()

# Inicializa un contador para los valores numéricos encontrados al inicio
contador_numeros_al_inicio = 0

# Función para determinar si una cadena es numérica
def es_numerico(cadena):
    try:
        float(cadena)  # Intenta convertir la cadena a un valor flotante
        return True
    except ValueError:
        return False

# Abre el archivo para leer y busca valores numéricos al inicio de las líneas
try:
    with open(ruta_archivo, 'r', encoding='utf-8') as archivo, open(ruta_archivo_salida, 'w', encoding='utf-8') as archivo_salida:
        for linea in archivo:
            palabra = linea.strip().split(' ')[0]  # Extrae el primer elemento de cada línea
            if es_numerico(palabra):
                archivo_salida.write(palabra + '\n')  # Escribe el número en el archivo de salida
                contador_numeros_al_inicio += 1

except Exception as e:
    print(f"Ocurrió un error al procesar el archivo: {e}")

# Finaliza el tiempo de ejecución
final = time.time()

# Muestra la cantidad de "palabras" numéricas encontradas y el tiempo total de ejecución
print(f"'Palabras' numéricas encontradas al inicio y guardadas: {contador_numeros_al_inicio}")
print(f"Tiempo total de ejecución: {final - inicio:.2f} segundos.")
