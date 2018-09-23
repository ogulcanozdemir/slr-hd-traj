import re
import numpy as np


def parse_output(output_file):
    with open(output_file, 'r') as file:
        output = file.read()

    # line_pattern = r"(?P<eta>\d+\.\d+)"
    line_pattern = r"(?P<eta>\d+) milliseconds"

    eta = []
    for r in re.findall(line_pattern, output):
        eta.append(float(r))

    return np.mean(eta, axis=0) / 1000.0


filename = 'results.txt'
print(parse_output(filename))
