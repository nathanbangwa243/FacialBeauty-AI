import shutil
import os
import numpy as np

FKP_MODELS = np.array([49, 55, 28, 32, 36, 4, 14])

SAFE_MODEL = os.path.join(os.getcwd(), "safeModel", "naimish.h5")

DEST = os.path.join(os.getcwd(), "api", "models")


def main():
    for fkp in FKP_MODELS:
        # copy naimish to API/MODELS
        # oldModelFile = os.path.join(DEST, f'naimish.h5')
        newModelFile = os.path.join(DEST, f'FKP{fkp}.h5')

        # rename
        # os.rename(oldModelFile, newModelFile)

        shutil.copy(SAFE_MODEL, newModelFile)

        print(f"okay : {newModelFile}")

if __name__ == "__main__":
    main()