from getopt import getopt
from pathlib import Path, PureWindowsPath


def parse_args(argvs):
    opts, arguments = getopt(argvs, "t:d:r:m:c:b:e:n:l:p:w",
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
        "use-weighted-loss"
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
        else:
            raise ValueError(f"Option {o} is not recognized.")
    return output