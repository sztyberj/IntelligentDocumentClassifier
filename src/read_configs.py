import yaml

def read_config(path):
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    read_config()