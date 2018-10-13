import os
import re
import json

can_dir = './predict/candidate/'
ref_dir = './predict/reference/'
word_dict_file = '../test_label_dict.json'

with open(word_dict_file, 'r') as f:
    word_dict = json.load(f)

special_num = 5  # ZSL类别的个数
vocab_size = len(word_dict)

can = []
ref = []

can_normal = []
ref_normal = []
for i in range(vocab_size - special_num):
    can_normal.append([])
    ref_normal.append([])

can_special = []
ref_special = []
for i in range(special_num):
    can_special.append([])
    ref_special.append([])

special_correct_all = 0
normal_correct_all = 0
can_normal_num = 0
ref_normal_num = 0
can_special_num = 0
ref_special_num = 0

can_stat = [0] * (len(word_dict) + 1)
ref_stat = [0] * (len(word_dict) + 1)
can_all = 0
ref_all = 0


for can_file in os.listdir(can_dir):
    i = int(re.findall('\d+', can_file)[0]) # 文件id
    with open(can_dir + can_file, 'r') as f:
        line = f.readlines()[0].strip().split()
        for ii, label in enumerate(line):
            line[ii] += ' '
        for label in line:
            # if word_dict.get(label, -1) == -1:
                # print('{}|'.format(label))
                # print(line)
                # print(can_file)
            idx = word_dict.get(label, vocab_size)
            if idx > vocab_size - special_num - 1 and idx < vocab_size:
                can_special[vocab_size - 1 - idx].append(i)
                can_special_num += 1
            elif idx < vocab_size:
                if i in can_normal[idx]:
                    print('duplicate here, line: {}'.format(i))
                can_normal[idx].append(i)
                can_normal_num += 1
            can_stat[idx] += 1
            can_all += 1
        can.append(line)
can_avg = can_all / len(os.listdir(can_dir))

for ref_file in os.listdir(ref_dir):
    i = int(re.findall('\d+', ref_file)[0]) # 文件id
    with open(ref_dir + ref_file, 'r') as f:
        line = f.readlines()[0].strip().split()
        for ii, label in enumerate(line):
            line[ii] += ' '
        for label in line:
            idx = word_dict.get(label, vocab_size)
            if idx > vocab_size - special_num - 1 and idx < vocab_size:
                ref_special[vocab_size - idx - 1].append(i)
                ref_special_num += 1
            elif idx < vocab_size:
                ref_normal[idx].append(i)
                ref_normal_num += 1
            ref_stat[idx] += 1
            ref_all += 1
        ref.append(line)
ref_avg = ref_all / len(os.listdir(ref_dir))

can_special.reverse()
ref_special.reverse()  # 顺序是反的

for i, id_set in enumerate(can_normal):
    correct = 0
    for idx in id_set:
        if idx in ref_normal[i]:
            correct += 1
    normal_correct_all += correct
    print('can/ref: {}/{}/{}'.format(correct, len(can_normal[i]), ref_stat[i]))

print('normal acc: {}'.format(normal_correct_all / can_normal_num))
print('can/ref: {}/{}'.format(can_normal_num, ref_normal_num))

for i, id_set in enumerate(can_special):
    correct = 0
    for idx in id_set:
        if idx in ref_special[i]:
            correct += 1
    special_correct_all += correct
    print('can/ref(special): {}/{}/{}'.format(correct, len(can_special[i]), len(ref_special[i])))

if can_special_num:
    print("special acc: {}".format(special_correct_all / can_special_num))
    print("can/ref: {}/{}".format(can_special_num, ref_special_num))
else:
    print('no special')

with open("candidate_aggregate.txt", 'w') as f:
    # restore str from list
    for i, line in enumerate(can):
        can[i] = ' '.join(line) + '\n'
    f.writelines(can)

with open("reference_aggregate.txt", 'w') as f:
    # restore str from list
    for i, line in enumerate(ref):
        ref[i] = ' '.join(line) + '\n'

    f.writelines(ref)

print (can_stat, ref_stat)
print("can_avg: {}, ref_avg: {}".format(can_avg, ref_avg))


