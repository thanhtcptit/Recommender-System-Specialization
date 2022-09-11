from collections import defaultdict

import numpy as np


def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


user_list = []
user_ratings = defaultdict(lambda: [])
mov_ratings = defaultdict(lambda: [])
mov_ratings_norm = defaultdict(lambda: [])
with open("ratings.csv") as f:
    mov_list = f.readline().strip().split(",")[1:-1]
    for line in f:
        d = line.strip().split(",")
        user_list.append(d[0])
        ratings = [float(r) if r else 0 for r in d[1:-1]]
        mean_rating = np.mean([i for i in ratings if i != 0])
        user_ratings[d[0]] = ratings
        for i, mi in enumerate(mov_list):
            mov_ratings[mi].append(ratings[i])
            mov_ratings_norm[mi].append(ratings[i] - mean_rating if ratings[i] else 0)

mov_cosine_similarity, mov_norm_cosine_similarity = defaultdict(lambda: {}), defaultdict(lambda: {})
top5_mov_similarity, top5_mov_norm_similarity = defaultdict(lambda: {}), defaultdict(lambda: {})
for i in mov_list:
    for j in mov_list:
        mov_cosine_similarity[i][j] = cosine_similarity(mov_ratings[i], mov_ratings[j])
        mov_norm_cosine_similarity[i][j] = cosine_similarity(mov_ratings_norm[i], mov_ratings_norm[j])

    top5_sorted = sorted(mov_cosine_similarity[i].items(), key=lambda x: x[1], reverse=True)[:6]
    top5_mov_similarity[i] = {x[0]: x[1] for x in top5_sorted}

    top5_norm_sorted = sorted(mov_norm_cosine_similarity[i].items(), key=lambda x: x[1], reverse=True)[:6]
    top5_mov_norm_similarity[i] = {x[0]: x[1] for x in top5_norm_sorted}

top5_mov_similarity['1: Toy Story (1995)']
top5_mov_norm_similarity['1: Toy Story (1995)']

predict_user = "5277"
mov_pred_score = {}
mov_norm_pred_score = {}
for pi in mov_list:
    a = b = c = d = 0
    for i, mi in enumerate(mov_list):
        if mov_cosine_similarity[pi][mi] <= 0 or user_ratings[predict_user][i] == 0:
            continue
        a += mov_cosine_similarity[pi][mi] * user_ratings[predict_user][i]
        b += mov_cosine_similarity[pi][mi]

        if mov_norm_cosine_similarity[pi][mi] > 0:
            c += mov_norm_cosine_similarity[pi][mi] * user_ratings[predict_user][i]
            d += mov_norm_cosine_similarity[pi][mi]
    if b != 0:
        pr = a / b
    else:
        pr = 0
    if d != 0:
        norm_pr = c / d
    else:
        norm_pr = 0
    mov_pred_score[pi] = pr
    mov_norm_pred_score[pi] = norm_pr

top5_predict_mov = sorted(mov_pred_score.items(), key=lambda x: x[1], reverse=True)[:5]
top5_predict_norm_mov = sorted(mov_norm_pred_score.items(), key=lambda x: x[1], reverse=True)[:5]

top5_predict_mov
top5_predict_norm_mov