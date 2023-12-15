from pathlib import Path

import yaml

def load(filepath:Path, max_duration=None):
  with filepath.open("r") as fp:
    config = yaml.safe_load(fp)

    yaml.Dumper.ignore_aliases = lambda *args: True
    config = yaml.safe_load(yaml.dump(config))

    return config
