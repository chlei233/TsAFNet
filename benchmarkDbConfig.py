import os
import glob
def parse_ucf_splits():
    class_ind = [x.strip().split() for x in open('data/ucf101/ucfTrainTestlist/classInd.txt')]
    #print(class_ind)
    #global class_mapping
    class_mapping = {x[1]:int(x[0])-1 for x in class_ind}

    def line2rec(line):
        items = line.strip().split('/')
        label = class_mapping[items[0]]
        vid = items[1].split('.')[0]
        return vid, label

    splits = []
    # 得到三个(train_list, test_list),  其中每个list，包含一整个文件的数据，格式为[(vid, label), ()...]
    for i in range(1, 4):
        train_list = [line2rec(x) for x in open('data/ucf101/ucfTrainTestlist/trainlist{:02d}.txt'.format(i))]
        test_list = [line2rec(x) for x in open('data/ucf101/ucfTrainTestlist/testlist{:02d}.txt'.format(i))]
        splits.append((train_list, test_list))
    return splits, class_mapping

def parse_hmdb51_splits():
    # load split file
    class_files = glob.glob('data/HDMB51/HDMB51testTrainMulti_7030_splits/*split*.txt')

    # load class list
    class_list = [x.strip() for x in open('data/HDMB51/HDMB51testTrainMulti_7030_splits/class_list.txt')]
    class_dict = {x: i for i, x in enumerate(class_list)}

    def parse_class_file(filename):
        # parse filename parts  
        filename_parts = filename.split('/')[-1][:-4].split('_')
        split_id = int(filename_parts[-1][-1])
        class_name = '_'.join(filename_parts[:-2])

        # parse class file contents
        contents = [x.strip().split() for x in open(filename).readlines()]
        train_videos = [ln[0][:-4] for ln in contents if ln[1] == '1']
        test_videos = [ln[0][:-4] for ln in contents if ln[1] == '2']

        return class_name, split_id, train_videos, test_videos

    #class_info_list = map(parse_class_file, class_files)   py3迭代器只能迭代一次，直接list化可多次迭代
    class_info_list = list(map(parse_class_file, class_files))

    splits = []
    for i in range(1, 4):
        train_list = [
            (vid, class_dict[cls[0]]) for cls in class_info_list for vid in cls[2] if cls[1] == i
        ]
        test_list = [
            (vid, class_dict[cls[0]]) for cls in class_info_list for vid in cls[3] if cls[1] == i
        ]
        splits.append((train_list, test_list))
    return splits, class_dict

hmdbFrames_path = './data/HDMB51Frames'
ucfFrames_path = './data/ucf101Frames'

hmdb_splits_tp, class_mapping_hmdb = parse_hmdb51_splits()
class_mapping_hmdb = {v:k for k,v in class_mapping_hmdb.items()}
ucf_splits_tp, class_mapping_ucf = parse_ucf_splits()
class_mapping_ucf = {v:k for k,v in class_mapping_ucf.items()}
#print("length of ucf_split_train: {}".format(len(ucf_splits_tp[0][0])))
#print("length of hmdb_split_train: {}".format(len(hmdb_splits_tp[0][0])))
def formal_split_hmdb():
    trainTest_split = []
    # train_split
    #train_split = []
    #test_split = []
    for i in range(len(hmdb_splits_tp)):
        train_split_tp = []
        test_split_tp = []
        for x in hmdb_splits_tp[i][0]:
            vid, label = x
            vidFrames_path = os.path.join(hmdbFrames_path, class_mapping_hmdb[label], vid) 
            frames_num = len(os.listdir(vidFrames_path)) 
            train_split_tp.append((vidFrames_path, frames_num, label))
        #train_split.append(train_split_tp)
        for x in hmdb_splits_tp[i][1]:
            vid, label = x
            vidFrames_path = os.path.join(hmdbFrames_path, class_mapping_hmdb[label], vid) 
            frames_num = len(os.listdir(vidFrames_path)) 
            test_split_tp.append((vidFrames_path, frames_num, label))
        #test_split.append(test_split_tp)  

        trainTest_split.append((train_split_tp, test_split_tp))  

    return trainTest_split

def formal_split_ucf():
    trainTest_split = []
    # train_split
    #train_split = []
    #test_split = []
    for i in range(len(ucf_splits_tp)):
        train_split_tp = []
        test_split_tp = []
        for x in ucf_splits_tp[i][0]:
            vid, label = x
            vidFrames_path = os.path.join(ucfFrames_path, class_mapping_ucf[label], vid) 
            frames_num = len(os.listdir(vidFrames_path)) 
            train_split_tp.append((vidFrames_path, frames_num, label))
        #train_split.append(train_split_tp)
        for x in ucf_splits_tp[i][1]:
            vid, label = x
            vidFrames_path = os.path.join(ucfFrames_path, class_mapping_ucf[label], vid) 
            frames_num = len(os.listdir(vidFrames_path)) 
            test_split_tp.append((vidFrames_path, frames_num, label))
        #test_split.append(test_split_tp)  

        trainTest_split.append((train_split_tp, test_split_tp))  

    return trainTest_split

#x = formal_split_hmdb()

#dataset = 'HDMB51'
#out_path = './data'
#print("length of x: {}".format(len(x)), len(x[0]), len(x[0][0]), x[0][0][0])
#open(os.path.join(out_path, '{}_rgb_train_split_{}.txt'.format(dataset, 1)), 'w').writelines(("".join(lists[0][0])))