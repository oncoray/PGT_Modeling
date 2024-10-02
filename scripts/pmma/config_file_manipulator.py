#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 10:42:16 2023

@author: kiesli21
"""
import yaml
import re
import os
import json
    
def save_config(config, filename):
    """
    Save the configuration dictionary to a YAML file.

    Args:
    config (dict): The configuration dictionary to save.
    filename (str): The path to the file where to save the configuration.
    """
    directory = os.path.dirname(filename)
    os.makedirs(directory, exist_ok=True)
    
    with open(filename, 'w') as f:
        yaml.dump(config, f)
    print("Config file saved!")

def generate_context(node, context={}, path=[]):
    """
    Generate context dictionary from YAML configuration.
    """
    if isinstance(node, dict):
        for k, v in node.items():
            context = generate_context(v, context, [k])  # We keep only the last key
    elif isinstance(node, list):
        for i, v in enumerate(node):
            context = generate_context(v, context, path)
    else:
        context[path[0]] = node
    return context

def substitute_values(node, context):
    """
    Substitute placeholders in YAML configuration.
    """
    
    if isinstance(node, dict):
        return {k: substitute_values(v, context) for k, v in node.items()}
    elif isinstance(node, list):
        return [substitute_values(v, context) for v in node]
    elif isinstance(node, str):
        # Find all placeholders
        placeholders = re.findall(r'\{(.+?)\}', node)

        # Substitute each placeholder
        for placeholder in placeholders:
            # Extract keys from placeholder
            keys = placeholder.split('.')
            # Check if last key in placeholder exists in context
            if keys[-1] in context:
                node = node.replace(f'{{{placeholder}}}', str(context[keys[-1]]))
            else:
                raise KeyError(f"Label '{placeholder}' not defined in configuration file.")

        return node
    else:
        return node

def substitute_labels(filename):
    """
    Substitute bracketed labels in YAML file based on the keys defined within the YAML.

    Args:
    filename (str): YAML file to process.

    Returns:
    dict: YAML contents with substitutions made.
    """
    print("Substitute name placeholders in config file....")
    # Load YAML file
    with open(filename, 'r') as f:
        data = yaml.safe_load(f)

    # Generate context dictionary
    context = generate_context(data)

    # Replace placeholders with their respective values
    data = substitute_values(data, context)

    return data


def yaml_to_json_string(yaml_string):
    """
    Convert a string in YAML format to a JSON-formatted string.

    The function takes a YAML-formatted string, loads it into a Python 
    dictionary using PyYAML's safe_load() function, then converts that 
    dictionary into a JSON-formatted string using json.dumps().

    Args:
        yaml_string (str): A string containing a dictionary in YAML format.

    Returns:
        str: A JSON-formatted string equivalent of the input.

    Raises:
        YAMLError: If there is an error parsing the YAML string.
    """
    try:
        # Convert the YAML string to a Python dictionary
        data = yaml.safe_load(yaml_string)
    except yaml.YAMLError as exc:
        print(exc)

    # Convert the Python dictionary to a JSON string
    return json.dumps(data)

