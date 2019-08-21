
import numpy as np
import json, codecs

def weights2json(weights):
    """
    Convert weights into json string
    :param weights: list of numpy arrays
    :return: json string
    """
    python_format = [w.tolist() for w in weights]
    return json.dumps(python_format,separators=(',', ':'), sort_keys=True, indent=4)


def jsontoweights(json_string):
    """
    Load list of weights from json
    :param json: json string
    :return: list of numpy arrays
    """
    python_format = json.loads(json_string)

    return [np.array(w) for w in python_format]
