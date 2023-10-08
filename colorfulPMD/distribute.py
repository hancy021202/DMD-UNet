import os
from shutil import copy, rmtree
import random


def mk_file(file_path: str):
    if os.path.exists(file_path):
        # 如果文件夹存在，则先删除原文件夹在重新创建
        rmtree(file_path)
    os.makedirs(file_path)

#######################################没写好！！！！！！！！！！！！！！！！！！！！
def main():
    # 保证随机可复现
    random.seed(1)

    # 将数据集中20%的数据划分到验证集中
    validate_rate = 0.2

    # 指向你解压后的flower_photos文件夹
    cwd = os.getcwd()
    data_root = os.path.join(cwd, "data")
    origin_data_path = os.path.join(data_root, "data_new")
    assert os.path.exists(origin_data_path), "path '{}' does not exist.".format(origin_data_path)

    flower_class = [cla for cla in os.listdir(origin_data_path)
                    if os.path.isdir(os.path.join(origin_data_path, cla))]

    # 建立保存训练集的文件夹
    train_root = os.path.join(data_root, "data_new", "train_new.py")
    mk_file(train_root)
    # for cla in flower_class:
    #     # 建立每个类别对应的文件夹
    #     mk_file(os.path.join(train_root, cla))

    # 建立保存验证集的文件夹
    val_root = os.path.join(data_root, "data_new",  "val_new")
    mk_file(val_root)
    # for cla in flower_class:
    #     # 建立每个类别对应的文件夹
    #     mk_file(os.path.join(val_root, cla))

    #######################################没写好！！！！！！！！！！！！！！！！！！！！

    for cla in flower_class:
        cla_path = os.path.join(origin_data_path, cla)
        data = os.listdir(cla_path)
        num = len(data)
        # 随机采样验证集的索引
        eval_index = random.sample(data, k=int(num*validate_rate))
        for index, data in enumerate(data):
            if data in eval_index:
                # 将分配至验证集中的文件复制到相应目录
                image_path = os.path.join(cla_path, data)
                new_path = os.path.join(val_root, cla)
                copy(image_path, new_path)
            else:
                # 将分配至训练集中的文件复制到相应目录
                image_path = os.path.join(cla_path, data)
                new_path = os.path.join(train_root, cla)
                copy(image_path, new_path)
            print("\r[{}] processing [{}/{}]".format(cla, index+1, num), end="")  # processing bar
        print()

    print("processing done!")


if __name__ == '__main__':
    main()
