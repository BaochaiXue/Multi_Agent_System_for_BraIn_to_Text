"""YAML configuration with attribute access and ``key.subkey=value`` command-line overrides."""

from __future__ import annotations

import yaml


class Config(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    __setattr__ = dict.__setitem__


def to_config(value):
    if isinstance(value, dict):
        return Config({key: to_config(item) for key, item in value.items()})
    if isinstance(value, list):
        return [to_config(item) for item in value]
    return value


def to_plain(value):
    if isinstance(value, dict):
        return {key: to_plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain(item) for item in value]
    return value


def load_config(path: str, overrides: list[str] | None = None) -> Config:
    with open(path) as stream:
        raw = yaml.safe_load(stream)
    for override in overrides or []:
        key, value = override.split("=", 1)
        node = raw
        *parents, leaf = key.split(".")
        for parent in parents:
            node = node[parent]
        node[leaf] = yaml.safe_load(value)
    return to_config(raw)


def save_config(config: Config, path):
    with open(path, "w") as stream:
        yaml.safe_dump(to_plain(config), stream, sort_keys=False)
