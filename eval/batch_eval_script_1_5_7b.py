
cuda_id_list = [0, 1]

test_for_each_cuda = 4

final_all_log = 'total_log.csv'


log_root_folder = './eval/logs/llava1.5_7b/'
model_args_str = 'pretrained=llava-hf/llava-1.5-7b-hf,device_map=auto,attn_implementation=eager'

Start_Time_String = "09:00:00"
duration_string = "48:00:00"


# ===================================
# Scheduled start settings
# ===================================

import time
from datetime import datetime, timedelta
from loguru import logger

def next_time_stamp(start_time_string):
    now = datetime.now()
    start_time = datetime.strptime(start_time_string, "%H:%M:%S").time()
    today_start_time = now.replace(hour=start_time.hour, minute=start_time.minute, second=start_time.second, microsecond=0)
    if now >= today_start_time:
        next_start_time = today_start_time + timedelta(days=1)
    else:
        next_start_time = today_start_time
    next_start_time_timestamp = next_start_time.timestamp()
    return next_start_time_timestamp

def timestamp_after_duration(initial_timestamp, duration_string):
    initial_time = datetime.fromtimestamp(initial_timestamp)
    duration_parts = duration_string.split(":")
    hours = int(duration_parts[0])
    minutes = int(duration_parts[1])
    seconds = int(duration_parts[2])
    duration = timedelta(hours=hours, minutes=minutes, seconds=seconds)
    new_time = initial_time + duration
    new_timestamp = new_time.timestamp()
    return new_timestamp

start_timestamp = next_time_stamp(Start_Time_String)
end_timestamp = timestamp_after_duration(start_timestamp, duration_string)

while time.time() < start_timestamp:
    time.sleep(10)

logger.info('Eval Program Start!')

# ===================================



import subprocess
import threading
import os

lock = threading.Lock()




tasks = ['docvqa', 'ocrbench', 'nocaps', 'textvqa']


config_list = [
    # Raw Model
    {
        'vision_adaptive_attention': ['false'],
        'vision_adaptive_attention_rate': [0.5],
        'vision_adaptive_attention_compute_rate': [0.7],
        'vision_adaptive_attention_update_interval': [3],
        'text_pruning_flag': ['false'],
        'text_pruning_rate': [0.4],
        'prefill_filter_flag': ['false'],
        'prefill_filter_rate': [0.5],
    },

]






import queue



job_queue = queue.Queue()


from copy import deepcopy

def get_job_dict(keys_set, config_dict, job_dict):

    job_dict = deepcopy(job_dict)

    if len(keys_set) == 0:

        global job_queue

        job_queue.put(job_dict)

    else:

        keys_set = deepcopy(keys_set)

        key = keys_set.pop()

        value_list = config_dict[key]

        for value in value_list:

            job_dict[key] = value

            get_job_dict(keys_set, config_dict, deepcopy(job_dict))







for task in tasks:

    for a_config in config_list:

        keys_set = set(a_config.keys())

        default_dict = {
            'tasks': task
        }

        get_job_dict(keys_set, a_config, default_dict)




logger.info(f'Job Loaded. Job num: {job_queue.qsize()}')

class Machine():

    def __init__(self, cuda_id):
        self.is_run = False
        self.cuda_id = cuda_id


machine_list = []

for i in range(test_for_each_cuda):
    for cuda_id in cuda_id_list:
        machine_list.append(Machine(cuda_id=cuda_id))

logger.info(f'Machine Loaded. machine num: {len(machine_list)}')


import csv



def write_to_file(results_data, job_dict):

    lock.acquire()

    job_dict_key = list(job_dict.keys())
    job_dict_key.sort()

    headers = ['task_detail', 'results'] + job_dict_key

    final_log_path = os.path.join(log_root_folder, final_all_log)

    if os.path.exists(final_log_path) == False:
        with open(final_log_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()

    input_dict_list = []

    for key in results_data.keys():
        item = {
            'task_detail': key,
            'results': str(results_data[key])
        }
        item.update(job_dict)
        input_dict_list.append(item)

    try:
        with open(final_log_path, 'a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=headers)

            for row in input_dict_list:
                writer.writerow(row)

    finally:
        lock.release()



import os

exit_all_job = False


def get_cmd(job_dict):

    log_folder = log_root_folder + job_dict["tasks"]

    os.makedirs(log_folder, exist_ok=True)

    cmd = ["/opt/conda/envs/aaai2025/bin/python", './src/lmms-eval/lmms_eval/__main__.py', 
           '--model', 'llava_hf', '--model_args', model_args_str,
           '--batch_size', '1',
           '--log_samples',
           '--output_path', log_folder
           ]
    
    
    log_samples_suffix = model_args_str[24:39]
    
    cmd.append('--log_samples_suffix')
    cmd.append(log_samples_suffix)
    
    for key in job_dict.keys():
        cmd.append(f'--{key}')
        cmd.append(f'{job_dict[key]}')

    return cmd, log_folder



def run_job(job_dict, machine_idx):


    cmd, log_folder = get_cmd(job_dict)

    global machine_list

    machine_list[machine_idx].is_run = True

    cuda_id = machine_list[machine_idx].cuda_id
    
    logger.info(f'CUDA: {cuda_id}, CMD: {str(cmd)}')

    env = os.environ.copy()

    env['CUDA_VISIBLE_DEVICES'] = str(cuda_id)

    process = subprocess.Popen(cmd, env=env)

    while True:

        time.sleep(4)

        if process.poll() is not None:
            break

        global exit_all_job

        if exit_all_job:
            process.terminate()
            time.sleep(10)
            process.kill()
            process.wait()
            return
        
    machine_list[machine_idx].is_run = False




    # write log

    cmd[1] = "./src/lmms-eval/lmms_eval/get_args.py"

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    suffix = '[args.a_vl_hash:'
    index = result.stdout.find(suffix)
    
    if index == -1:
        return None
    
    index += len(suffix)

    hash_str = ''

    while True:

        hash_str += result.stdout[index]
        index += 1

        if result.stdout[index] == ']':
            break

    logger.debug(f'Get Hash Code: {hash_str}')

    files = os.listdir(log_folder)

    output_log_folder = None

    for file in files:
        if file[-len(hash_str):] == hash_str:
            output_log_folder = file
            break

    if output_log_folder is None:
        return None

    output_log_folder = os.path.join(log_folder, output_log_folder)
    output_log_result_file_path = os.path.join(output_log_folder, 'results.json')

    import json
    with open(output_log_result_file_path, 'r') as file:
        data = json.load(file)
        results_data = data["results"]

    job_dict['result_final_output_to'] = output_log_folder

    write_to_file(results_data, job_dict)




def job_finished(job_dict):

    cmd, log_folder = get_cmd(job_dict)

    cmd[1] = "./src/lmms-eval/lmms_eval/get_args.py"

    result = subprocess.run(cmd, capture_output=True, text=True)

    suffix = '[args.a_vl_hash:'
    index = result.stdout.find(suffix)
    
    assert index > -1
        
    
    index += len(suffix)

    hash_str = ''

    while True:

        hash_str += result.stdout[index]
        index += 1

        if result.stdout[index] == ']':
            break

    files = os.listdir(log_folder)

    for file in files:
        if file[-len(hash_str):] == hash_str:
            return True


    return False




import time

job_threads_list = []

machine_idx = -1

while not job_queue.empty():

    machine_idx = (machine_idx + 1) % len(machine_list)

    if machine_idx == 0:

        if time.time() >= end_timestamp:

            logger.warning('TIMEOUT! SYSTEM QUIT!')
            exit_all_job = True

            break

        time.sleep(10)

    if machine_list[machine_idx].is_run == False:

        if job_queue.empty():
            break

        while True:

            job_dict = job_queue.get()

            if not job_finished(job_dict):
                break


        thread = threading.Thread(target=run_job, args=(job_dict, machine_idx,))

        thread.start()

        job_threads_list.append(thread)



for thread in job_threads_list:
    thread.join()



