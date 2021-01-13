# import trainingFB
import sys


def main():
    # args
    assert len(sys.argv) > 1

    # target fkp
    targetFKP = int(sys.argv[1])

    assert targetFKP in trainingFB.config.TARGET_FKPS

    # train model
    trainingFB.train.trainModel(targetFKP)


if __name__ == "__main__":
    main()
