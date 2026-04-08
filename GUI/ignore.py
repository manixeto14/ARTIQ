import h5py
import numpy as np
import re

# Configuración
file_path = r'Data\000001190-ExperimentalControl\000001190-ExperimentalControl.h5'
# Generamos los 50 valores
valores_linspace = []
for i in range(50):
    valores_linspace.append(i//10)

try:
    with h5py.File(file_path, 'r+') as f:
        datasets_group = f['datasets']

        for name in datasets_group.keys():
            # Buscamos carpetas que sigan el patrón "Scan_X"
            match = re.match(r'Scan_(\+?\d+)', name)

            if match:
                # Extraemos el número del scan
                scan_number = int(match.group(1))

                # El índice para el array (Scan_1 -> índice 0, Scan_2 -> índice 1...)
                idx = scan_number - 1

                # Verificamos que el índice esté dentro del rango de nuestro linspace (0 a 49)
                if 0 <= idx < len(valores_linspace):
                    valor_a_asignar = valores_linspace[idx]

                    path_to_param = f'datasets/{name}/Parameters/Parameter1'

                    if path_to_param in f:
                        # Eliminamos y creamos el dataset con el valor único escalar
                        del f[path_to_param]
                        # Guardamos el valor como un array de un solo elemento o un escalar
                        f.create_dataset(path_to_param, data=np.array(valor_a_asignar))

                        print(f":white_check_mark: {name} actualizado con valor: {valor_a_asignar:.4f} (índice {idx})")
                    else:
                        print(f":warning: Parámetro no encontrado en {name}")
                        f.create_dataset(path_to_param, data=np.array(valor_a_asignar))
                else:
                    print(f":information_source: {name} fuera de rango para el linspace de 50 elementos.")

    print("\n:sparkles: Proceso completado.")

except Exception as e:
    print(f":x: Error: {e}")