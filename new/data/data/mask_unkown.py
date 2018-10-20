file = './train_label.txt'

new_label = ['bua', 'tlk', 'zah', 'sko', 'buk']

new_lines = []
with open(file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        labels = line.strip().split()
        flag = 0
        for new in new_label:
            if new in labels:
                flag = 1
        if flag:
            labels = ['<unk>']
        new_lines += (' '.join(labels) + '\n')

with open(file, 'w') as f:
    f.writelines(new_lines)



