#  Copyright (c) 2019 École Polytechnique
# 
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this file, you can obtain one at http://mozilla.org/MPL/2.0
# 
#  Authors:
#        Luciano Di Palma <luciano.di-palma@polytechnique.edu>
#        Enhui Huang <enhui.huang@polytechnique.edu>
# 
#  Description:
#  AIDEme is a large-scale interactive data exploration system that is cast in a principled active learning (AL) framework: in this context,
#  we consider the data content as a large set of records in a data source, and the user is interested in some of them but not all.
#  In the data exploration process, the system allows the user to label a record as “interesting” or “not interesting” in each iteration,
#  so that it can construct an increasingly-more-accurate model of the user interest. Active learning techniques are employed to select
#  a new record from the unlabeled data source in each iteration for the user to label next in order to improve the model accuracy.
#  Upon convergence, the model is run through the entire data source to retrieve all relevant records.
from __future__ import annotations

import os
from typing import Optional, TYPE_CHECKING

import yaml

from definitions import RESOURCES_DIR

if TYPE_CHECKING:
    from aideme.utils.types import Config


def get_config_from_resources(resource: str, section: Optional[str] = None) -> Config:
    """
    Read an specific section from a YAML configuration file. Will throw exception is resource file does not exist.

    :param resource: resource file to read
    :param section: section of resource to read. If None, the entire resource will be returned
    :return: configuration as dict
    """
    path = get_path_to_resource(resource)

    with open(path, 'r') as file:
        conf = yaml.safe_load(file)
        return conf[section] if section else conf


def write_config_to_resources(resource: str,  section: str, config: Config) -> None:
    """
    Updates resources with a new config. Will throw exception is resource file does not exist.
    :param resource: name of resource to update
    :param section: name of section for new config
    :param config: configuration to write
    """
    resource_config = get_config_from_resources(resource)
    resource_config[section] = config

    path = get_path_to_resource(resource)

    with open(path, 'w') as file:
        yaml.dump(resource_config, file, Dumper=MyDumper, sort_keys=False, width=2000)


def get_path_to_resource(resource: str) -> str:
    path = os.path.join(RESOURCES_DIR, resource + '.yaml')

    if not os.path.exists(path):
        raise FileNotFoundError("Resource file {}.yaml does not exist.".format(resource))

    return path


class MyDumper(yaml.SafeDumper):
    # HACK: insert blank lines between top-level objects
    # inspired by https://stackoverflow.com/a/44284819/3786245
    def write_line_break(self, data=None):
        super().write_line_break(data)

        if len(self.indents) == 1:
            super().write_line_break()

    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)
