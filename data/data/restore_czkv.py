import json

with open('save_data.test.dict.trans.cz', 'r') as f:
    lines = f.readlines()
    lines = [line.strip() for line in lines]
    transcz = lines[4:]

with open('test_label_dict.json', 'r') as f:
    di = json.load(f)

for i, (k, v) in enumerate(di.items()):
    di[k] = transcz[i]

with open('label_cz.json', 'w') as f:
    json.dump(di, f)

