from collections import defaultdict

import numpy as np


def corr(x, y):
    a, b = [], []
    for i in range(len(x)):
        if x[i] != 0 and y[i] != 0:
            a.append(x[i])
            b.append(y[i])
    a, b = np.array(a), np.array(b)

    return np.sum((a - np.mean(a)) * (b - np.mean(b))) / \
        np.sqrt(np.sum((a - np.mean(a))  ** 2) * np.sum((b - np.mean(b)) ** 2))


def cvt_rating(x):
    if not x:
        x = 0
    return float(x)

mov_list = []
user_ratings = defaultdict(lambda: [])
mov_ratings = defaultdict(lambda: [])
with open("movie-row.csv") as f:
    user_ind = f.readline().strip().split(",")[1:]
    for line in f:
        d = line.strip().split(",")
        mov_list.append(d[0])
        ratings = [float(r) if r else 0 for r in d[1:]]
        mov_ratings[d[0]] = ratings
        for i, u in enumerate(user_ind):
            user_ratings[u].append(ratings[i])

user_mean_ratings = {}
for u, r in user_ratings.items():
    ratings = [x for x in r if x != 0]
    user_mean_ratings[u] = np.mean(ratings)

user_correlation, top5_user_correlation = defaultdict(lambda: {}), defaultdict(lambda: {})
for u1 in user_ind:
    for u2 in user_ind:
        if u1 == u2:
            continue
        user_correlation[u1][u2] = corr(user_ratings[u1], user_ratings[u2])
    top5_corr_sorted = sorted(user_correlation[u1].items(), key=lambda x: x[1], reverse=True)[:5]
    top5_user_correlation[u1] = {x[0]: x[1] for x in top5_corr_sorted}

predict_users = ["3712", "3867", "89"]
predict_ratings, predict_ratings_norm = defaultdict(lambda: []), defaultdict(lambda: [])
user_top5_predict_mov, user_top5_predict_norm_mov = {}, {}
for pu in predict_users:
    for mi in range(len(mov_list)):
        a = b = c = 0
        for u, cr in top5_user_correlation[pu].items():
            if user_ratings[u][mi] != 0:
                a += cr * user_ratings[u][mi]
                b += cr

                c += cr * (user_ratings[u][mi] - user_mean_ratings[u])
        if b != 0:
            pr = a / b
            norm_pr = user_mean_ratings[pu] + c / b
        else:
            pr = norm_pr = 0
        predict_ratings[pu].append(pr)
        predict_ratings_norm[pu].append(norm_pr)

    mov_rating_ind_sorted = np.argsort(predict_ratings[pu])[-5:][::-1]
    user_top5_predict_mov[pu] = {mov_list[i]: predict_ratings[pu][i] for i in mov_rating_ind_sorted}

    mov_rating_norm_ind_sorted = np.argsort(predict_ratings_norm[pu])[-5:][::-1]
    user_top5_predict_norm_mov[pu] = {mov_list[i]: predict_ratings_norm[pu][i] for i in mov_rating_norm_ind_sorted}
