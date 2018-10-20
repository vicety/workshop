1. Here we provide the RCV1-V2 and AAPD datasets, as well as pre-trained checkpoints of our SGM model and SGM+GE model on the RCV1-V2 dataset.

2. Json files in the folder 'RCV1-V2' and folder 'AAPD' are the constructed label dictionaries. The key is 
the name of the label, and the value is the corresponding index. Keys are sorted by frequency from highest 
to lowest. For example, the label with a value equal to 0 has the highest frequency of occurrence.

3. Each line in the label file corresponds to the true label sequence of each sample, and different labels
are separated by ' '.

4. If you use the AAPD dataset, please cite the following paper, and if there is any problem, please contact Pengcheng Yang (yang_pc@pku.edu.cn).

@inproceedings{YangCOLING2018,
   author = {Pengcheng Yang and Xu Sun and Wei Li and Shuming Ma and Wei Wu and Houfeng Wang},
   title = {SGM: Sequence Generation Model for Multi-label Classification},
   booktitle = {{COLING} 2018},
   year = {2018}
}