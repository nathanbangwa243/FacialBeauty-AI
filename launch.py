import os

from threading import Thread

# from trainingFB import config

import sys


# pythonEnv env
LOCAL_PYTHON_ENV = "C:\\ProgramData\\Anaconda3\\envs\\tensorflow\\python.exe"
AZURE_PYTHON_ENV = ""

# branch
TARGET_BRANCH = "training"


def pushToGithub():
    # ADD
    cmd = f"git add *"
    os.system(cmd)

    # COMMIT
    cmd = f'git commit -m "Training Model"'
    os.system(cmd)

    # PUSH
    cmd = f"git push origin {TARGET_BRANCH}"
    os.system(cmd)


def startTraining(pythonEnv, targetFkp):
    cmd = f"{pythonEnv} FacialBeauty.py {targetFkp}"
    os.system(cmd)



def main():
    # args
    assert len(sys.argv) > 1

    # target fkp
    interpreter = sys.argv[1].lower()

    assert interpreter in ['local', 'azure']

    if interpreter == 'local':
        pythonEnv = LOCAL_PYTHON_ENV
    else:
        pythonEnv = AZURE_PYTHON_ENV

    # create thread
    threads = []

    

    for fkp in [1, 2, 3]:#config.TARGET_FKPS:
        threads.append(
            Thread(
                target=startTraining,
                args=(pythonEnv, fkp),
                name=f"TrainingFKP#{fkp}"
            )
        )
    
    # launch
    for thread in threads:
        print(f"Starting of training {thread.name}")
        thread.start()
    
    # join
    for thread in threads:
        thread.join()
        print(f"End of training {thread.name}")

    
    # push to github
    print(f"Starting Push Process")

    pushToGithub()

    print(f"End of Push Process")



if __name__ == "__main__":
    main()