import os

def count_data(path):
    m = 0
    n = 0

    for name_dir in os.listdir(path):
        # skip non directory files
        if not os.path.isdir(os.path.join(path, name_dir)):
            continue

        for s in os.listdir(os.path.join(path, name_dir)):
            n += 1

        m += 1
        
    return m, n