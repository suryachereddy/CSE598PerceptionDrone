import yaml
import os


def scene_config(scene_id):
    with open(f'config{os.sep}scene{os.sep}{scene_id}.yml', 'r') as stream:
        try:
            print(yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)
