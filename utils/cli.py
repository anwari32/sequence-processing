from getopt import getopt
from pathlib import Path, PureWindowsPath


def parse_args(argvs):
    """
    return Object containing
    -   training-config
    -   device
    -   resume-run-ids
    -   model-config-dir
    -   model-config-names
    -   batch-size
    -   num-epochs
    -   run-name
    -   device-list
    -   project-name
    -   use-weighted-loss
    -   lr
    -   offline
    """
    opts, arguments = getopt(argvs, "t:d:r:m:c:b:e:n:l:p:w:",
    [
        "training-config=",
        "device=",
        "resume-run-ids=",
        "model-config-dir=",
        "model-config-names=",
        "batch-size=",
        "num-epochs=",
        "run-name=",
        "device-list=",
        "project-name=",
        "preprocessing-mode=",
        "accumulate-gradient",
        "use-weighted-loss",
        "lr=",
        "epsilon=",
        "beta1=",
        "beta2="
    ])
    output = {}
    for o, a in opts:
        if o in ["-t", "--training-config"]:
            output["training-config"] = str(Path(PureWindowsPath(a)))
        elif o in ["-d", "--device"]:
            output["device"] = a
        elif o in ["-r", "--resume-run-ids"]:
            output["resume-run-ids"] = a.split(',')
        elif o in ["-m", "--model-config-dir"]:
            output["model-config-dir"] = a
        elif o in ["-c", "--model-config-names"]:
            output["model-config-names"] = a.split(',')
        elif o in ["-b", "--batch-size"]:
            output["batch-size"] = int(a)
        elif o in ["-e", "--num-epochs"]:
            output["num-epochs"] = int(a)
        elif o in ["-n", "--run-name"]:
            output["run-name"] = a
        elif o in ["-l", "--device-list"]:
            output["device-list"] = [int(x) for x in a.split(',')]
        elif o in ["-p", "--project-name"]:
            output["project-name"] = a
        elif o in ["-w", "--use-weighted-loss"]:
            output["use-weighted-loss"] = True
        elif o in ["--baseline"]:
            output["model"] = "baseline"
        elif o in ["--sequence"]:
            output["model"] = "sequence"
        elif o in ["--whole"]:
            output["model"] = "recurrent"
        elif o in ["--offline"]:
            output["offline"] = True
        elif o in ["--accumulate-gradient"]:
            output["accumulate-gradient"] = True
        elif o in ["--lr"]:
            output["lr"] = float(a)
        elif o in ["--epsilon"]:
            output["epsilon"] = float(a)
        elif o in ["--beta1"]:
            output["beta1"] = float(a)
        elif o in ["--beta2"]:
            output["beta2"] = float(a)
        elif o in ["--preprocessing-mode"]:
            if a not in ["sparse", "dense"]:
                raise ValueError(f"Argument {o} accepts either `sparse` or `dense` value. Found {a}")
            output["preprocessing-mode"] = a
        else:
            raise ValueError(f"Option {o} is not recognized.")
    return output

class FineTuningConfig():
    def __init__(self, 
        name=None,
        pretrained=None,
        train_data=None,
        validation_data=None,
        test_data=None,
        freeze_bert=False,
        num_epochs=1,
        batch_size=1,
        learning_rate=1e-5,
        epsilon=1e-8,
        beta1=0.9,
        beta2=0.98,
        weight_decay=0.01
    ):
        self.name = name
        self.pretrained = pretrained
        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.freeze_bert = freeze_bert
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay

def parse_fine_tuning_command(argvs):
    opts, arguments = getopt(argvs, "t:c:d:",
    [
        "config-dir=",
        "config-names=",
        "device=",
        "device-list="
    ])

    output = {}
    for o, a in opts:
        if o in ["-t", "--config-dir"]:
            output["config-dir"] = str(a)
        elif o in ["-c", "--config-names"]:
            output["config-names"] = a.split(',')
        elif o in ["-d", "--device"]:
            output["device"] = a
        elif o in ["--device-list"]:
            output["device-list"] = a.split(",")
        else:
            raise ValueError(f"option {o} not recognized.")
    return output