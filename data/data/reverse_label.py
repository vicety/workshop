label_files = ['train_label.txt', 'dev_label.txt', 'test_label.txt']

for label_file in label_files:
    all_labels = []
    with open(label_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            labels = line.strip().split()
            labels.reverse()
            print(labels)
            labels = ' '.join(labels) + '\n'
            all_labels.append(labels)
    with open(label_file, 'w') as ff:
        ff.writelines(all_labels)

print("complete")
