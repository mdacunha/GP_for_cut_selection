import gzip
import shutil
import os



if __name__ == '__main__':
    zip_folder = "Instances_zip/"
    for file in os.listdir(zip_folder):
        if file.endswith('.mps.gz'):
            with gzip.open(zip_folder+file, 'rb') as f_in:
                with open("Instances_unziped/"+file[:len(file)-3], 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
