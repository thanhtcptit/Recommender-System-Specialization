from collections import defaultdict

import numpy as np
import pandas as pd


def corr(x, y):
    a, b = [], []
    for i in range(len(x)):
        if x[i] != 0 and y[i] != 0:
            a.append(x[i])
            b.append(y[i])
    a, b = np.array(a), np.array(b)

    return np.sum((a - np.mean(a)) * (b - np.mean(b))) / \
        np.sqrt(np.sum((a - np.mean(a))  ** 2) * np.sum((b - np.mean(b)) ** 2))


data = pd.read_csv("HW1-data.csv").fillna(0)
mov_list = data.columns[2:]
n_movies = len(mov_list)

mov_nrof_ratings = {}
mov_mean_ratings = {}
mov_percent_liking = {}
for i in range(2, n_movies + 2):
    ratings = [j for j in data.iloc[:, i] if j != 0]
    like_ratings = [j for j in ratings if j >= 4]
    mov_nrof_ratings[data.iloc[:, i].name] = len(ratings)
    mov_mean_ratings[data.iloc[:, i].name] = np.mean(ratings)
    mov_percent_liking[data.iloc[:, i].name] = len(like_ratings) / len(ratings)

mov_mean_ratings_sorted = sorted(mov_mean_ratings.items(), key=lambda x: x[1], reverse=True)
print("Quiz #1: ")
print(mov_mean_ratings_sorted[:3])

mov_nrof_ratings_sorted = sorted(mov_nrof_ratings.items(), key=lambda x: x[1], reverse=True)
print("Quiz #2: ")
print(mov_nrof_ratings_sorted[:3])

mov_percent_liking_sorted = sorted(mov_percent_liking.items(), key=lambda x: x[1], reverse=True)
print("Quiz #3: ")
print(mov_percent_liking_sorted[:3])

mov_associate_with_toy_story = defaultdict(lambda: 0)
for i, r in data.iterrows():
    if r.iloc[8] != 0:
        for j in range(2, n_movies + 2):
            if r.iloc[j] != 0:
                mov_associate_with_toy_story[r.index[j]] += 1
num_toy_story_ratings = sum([1 for i in data.iloc[:, 8] if i > 0])
mov_associate_with_toy_story = {k: v / num_toy_story_ratings for k, v in mov_associate_with_toy_story.items()}

mov_associate_with_toy_story_sorted = sorted(mov_associate_with_toy_story.items(), key=lambda x: x[1], reverse=True)
print("Quiz #4: ")
print(mov_associate_with_toy_story_sorted[:5])

mov_correlation = {}
for i in range(2, n_movies + 2):
    mov_correlation[data.iloc[:, i].name] = corr(data.iloc[:, i], data.iloc[:, 8])

mov_correlation_sorted = sorted(mov_correlation.items(), key=lambda x: x[1], reverse=True)
print("Quiz #5: ")
print(mov_correlation_sorted[:5])

mov_rating_female, mov_rating_male = defaultdict(lambda: []), defaultdict(lambda: [])
ratings_female, rating_male = [], []

mov_percent_liking_female, mov_percent_liking_male = defaultdict(lambda: []), defaultdict(lambda: [])
percent_liking_female, percent_liking_male = [], []
for i, r in data.iterrows():
    if r.iloc[1] == 0:
        for j in range(2, n_movies + 2):
            if r.iloc[j] != 0:
                mov_rating_male[r.index[j]].append(r.iloc[j])
                rating_male.append(r.iloc[j])

                mov_percent_liking_male[r.index[j]].append(int(r.iloc[j] >= 4))
                percent_liking_male.append(int(r.iloc[j] >= 4))
    else:
        for j in range(2, n_movies + 2):
            if r.iloc[j] != 0:
                mov_rating_female[r.index[j]].append(r.iloc[j])
                ratings_female.append(r.iloc[j])

                mov_percent_liking_female[r.index[j]].append(int(r.iloc[j] >= 4))
                percent_liking_female.append(int(r.iloc[j] >= 4))

mov_avg_rating_female = {k: np.mean(v) for k, v in mov_rating_female.items()}
mov_avg_rating_male = {k: np.mean(v) for k, v in mov_rating_male.items()}
avg_rating_diff = {}
for n in mov_list:
    avg_rating_diff[n] = mov_avg_rating_female[n] - mov_avg_rating_male[n]
avg_rating_diff_sorted = sorted(avg_rating_diff.items(), key=lambda x: x[1], reverse=True)
print("Quiz #6: ")
print(avg_rating_diff_sorted[0], avg_rating_diff_sorted[-1])
print(np.mean(ratings_female) - np.mean(rating_male))

mov_percent_liking_female = {k: np.mean(v) for k, v in mov_percent_liking_female.items()}
mov_percent_liking_male = {k: np.mean(v) for k, v in mov_percent_liking_male.items()}
percent_liking_diff = {}
for n in mov_list:
    percent_liking_diff[n] = mov_percent_liking_female[n] - mov_percent_liking_male[n]
percent_liking_diff_sorted = sorted(percent_liking_diff.items(), key=lambda x: x[1], reverse=True)
print("Quiz #7: ")
print(percent_liking_diff_sorted[0], percent_liking_diff_sorted[-1])
print(np.mean(percent_liking_female) - np.mean(percent_liking_male))
