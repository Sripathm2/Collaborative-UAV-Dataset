from random import shuffle
import os.path
import os
import time
import sys
from tqdm import tqdm

if __name__ == "__main__":

    files_to_ftp = []
    file_names = []
    file_size = []

    # get ip address of the server as cmd line argument
    ip = sys.argv[1]
    file_type = sys.argv[2]
    
    for path, subdirs, files in os.walk('/UAV_data/'):
        for name in files:
            file_name = os.path.join(path, name)
            if file_type not in file_name:
                continue
            file_names.append(name)
            file_size.append(os.path.getsize(file_name))
            file_name = file_name.replace('/UAV_data/','')
            files_to_ftp.append(file_name)
    

    shuffle(files_to_ftp)
    file_loop = 10
    while file_loop>0:
        for i in tqdm(range(len(files_to_ftp))):
        # for i in range(len(files_to_ftp)):
            file = files_to_ftp[i]
            start = time.time()
            os.system('wget -q http://' + ip + ':8888/' +file + ' > /dev/null 2>&1')
            end = time.time()
            os.system('rm -rf *.png *.mp4')
            print(ip + ' filename ' + file + ' time ' + str(end-start))
        print('All files recieved')
        file_loop -= 1
    print('FTP client done')
