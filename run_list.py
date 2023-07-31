import os

cmd_list = [
            "python train.py NAME \"CdC_alpha06lam07\" ALPHA 0.6 LAMBDA 0.7 > outputs\\test.txt", 
            "python train.py NAME \"CdC_alpha06lam10\" ALPHA 0.6 LAMBDA 1.0 > outputs\\test.txt"
            ]

for cmd in cmd_list:
    print(cmd)
    os.system(cmd)