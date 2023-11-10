from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import sys
import os
import subprocess
import time

base = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, base)

from knowledge.elasticsearch import ElasticSearchLibrary

index_name = "wiki"

lib = ElasticSearchLibrary(index_name)
dataset = load_dataset("graelo/wikipedia", "20230601.en")['train']
batch_size = 100

def get_cpu_temperature():
    try:
        command = "sensors | grep 'Core 0' | awk '{print $3}'"  # Esto puede variar según tu sistema
        temperature_str = subprocess.check_output(command, shell=True).decode().strip()
        temperature = float(temperature_str[:-2])  # Convierte la temperatura en un número flotante
        return temperature
    except Exception as e:
        print(f'No se pudo obtener la temperatura de la CPU: {str(e)}')
        return None  # Retorna None si no se puede obtener la temperatura

start = 0

if start == 0:
    lib.create_index()

temperauture = get_cpu_temperature()

print('Starting')

for i in range(start, len(dataset), batch_size):
    records = dataset[i:i + batch_size]

    payload = [{
        "id": int(records['id'][record]),
        "title": records['title'][record],
        "content": records['text'][record],
        "url": records['url'][record],
        "keyword": '',
        'command': ''
    } for record in range(len(records['id']))]

    lib.add(data_array=payload)
    temperauture = get_cpu_temperature()

    print('progress: ' + str(i) + ', ' + str(temperauture))

    time.sleep(1)

    if temperauture == None or temperauture > 85:
        time.sleep(60)
