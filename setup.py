import os

if __name__ == "__main__":
    for i in range(50):
        path = os.path.join("run", "seqlab-base-b64-small-e50-20220622-113606")
        dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        tdirs = [d.split(".")[0] for d in dirs]
        tdirs = [os.path.join(path, d) for d in tdirs]
        dirs = [os.path.join(path, d) for d in dirs]
        for src, dest in zip(dirs, tdirs):
            print(f"renaming {src} to {dest}")
            os.rename(src, dest)
