import time

# Define la ruta al archivo de vectores
ruta_archivo = 'C:\\Users\\afloresre\\Documents\\Cagh\\Red\\vectors400.txt'

# Inicia el tiempo de ejecución
inicio = time.time()

# Inicializa un contador para las palabras problemáticas
contador_palabras_problematicas = 0

try:
    # Abre el archivo para leer y establecer la longitud estándar de los vectores
    with open(ruta_archivo, 'r', encoding='utf-8') as archivo:
        # Obtiene la longitud de la primera línea para determinar la longitud estándar de los vectores
        primera_linea = archivo.readline().strip().split(' ')
        longitud_estandar = len(primera_linea) - 1

        # Restablece el puntero del archivo al inicio para revisar todas las líneas
        archivo.seek(0)

        # Itera sobre cada línea del archivo para verificar la longitud de los vectores
        for linea in archivo:
            elementos = linea.strip().split(' ')
            if len(elementos) - 1 != longitud_estandar:
                # Imprime la palabra inicial si la longitud del vector es incorrecta
                print(elementos[0])
                contador_palabras_problematicas += 1

except Exception as e:
    print(f"Ocurrió un error al procesar el archivo: {e}")

# Finaliza el tiempo de ejecución
final = time.time()

# Muestra la cantidad de palabras problemáticas encontradas y el tiempo total de ejecución
print(f"Palabras con longitud de vector incorrecta encontradas: {contador_palabras_problematicas}")
print(f"Tiempo total de ejecución: {final - inicio:.2f} segundos.")
