
with open('label_info', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    di = {}
    for line in lines[1:]:  # except root
        line = line.strip().split()
        child = line[3]
        desc = ' '.join(line[5:])
        desc = desc.replace('/', ' and ').lower()
        di[child] = desc
    print()