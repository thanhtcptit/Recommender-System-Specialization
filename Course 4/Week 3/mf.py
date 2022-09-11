import numpy as np


user_vectors, item_matrix, item_list, weights = {}, [], [], []

with open("user_matrix.csv") as f:
    f.readline()
    for line in f:
        d = line.strip().split("|")
        user_vectors[d[0]] = np.array(d[1:], dtype=np.float32)

with open("item_matrix.csv") as f:
    f.readline()
    d = f.readline().strip().split("|")
    weights = np.array(d[2:], dtype=np.float32)
    f.readline()
    for line in f:
        d = line.strip().split("|")
        item_list.append(d[0])
        item_matrix.append(np.array(d[2:], dtype=np.float32))
item_matrix = np.array(item_matrix)

top5_mov_f1_ind = np.argsort(item_matrix[:, 0])[::-1][:5]
for i in top5_mov_f1_ind:
    print(item_list[i], item_matrix[i, 0])

top5_mov_f2_ind = np.argsort(item_matrix[:, 1])[::-1][:5]
for i in top5_mov_f2_ind:
    print(item_list[i], item_matrix[i, 1])

score = np.dot(item_matrix, user_vectors["4469"] * weights)
top5_mov_ind = np.argsort(score)[::-1][:5]
for i in top5_mov_ind:
    print(item_list[i], score[i])