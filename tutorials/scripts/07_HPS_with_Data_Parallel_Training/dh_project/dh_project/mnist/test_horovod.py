import horovod.tensorflow.keras as hvd

from dh_project.mnist.model_run import run

def test():
    hvd.init()

    config = {
        "lr": 1e-3
    }

    score = run(config)
    if hvd.rank() == 0:
        print(f"Score: {score:.3f}")

if __name__ == "__main__":
    test()

