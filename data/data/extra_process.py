import re

_DIGIT_RE = re.compile(r"\d")

files = ['./dev_doc.txt', './train_doc.txt', './test_doc.txt']

for fil in files:
    with open(fil, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            lines[i] = re.sub(_DIGIT_RE, "0", line)
    with open(fil, 'w') as f:
        f.writelines(lines)
