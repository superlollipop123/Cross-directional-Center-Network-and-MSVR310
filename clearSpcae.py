import os

if __name__ == "__main__":
    clear_path_list = ['outputs_5', 'outputs_base']
    for path in clear_path_list:
        for f in os.listdir(path):
            f_path = os.path.join(path, f)
            if os.path.isdir(f_path):
                for file in os.listdir(f_path):
                    if file.endswith('pth'):
                        file_path = os.path.join(f_path, file)
                        os.remove(file_path)
                        print(file_path, 'removed')
        