#! /usr/bin/env python
from __future__ import print_function
import argparse
from collections import defaultdict
import numpy as np
import json
from tqdm import tqdm

# The reduce(fun,seq) function is used to apply a particular function passed in its argument to all of the list elements mentioned in the sequence passed along
from functools import reduce
import os
import operator
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from pathlib import Path
from math import sqrt
import random
from sklearn.metrics import precision_score, recall_score


TEST_IMAGES = "./src/video_node/images"

HIGHER_INLIERS = 0

top_n = 20
inliers_min = 0


def performance(predictions):
    gold = []
    with open("./src/video_node/ground_truth.txt") as f:
        for elem in f:
            gold.append(elem.strip())

    print("PRECISION")
    print("macro: ")
    print(precision_score(gold, predictions, average="macro"))
    print("\n")
    print("micro: ")
    print(precision_score(gold, predictions, average="micro"))
    print("\n")
    print("weighted: ")
    print(precision_score(gold, predictions, average="weighted"))
    print("\n")

    print("RECALL")
    print("macro: ")
    print(recall_score(gold, predictions, average="macro"))
    print("\n")
    print("micro: ")
    print(recall_score(gold, predictions, average="micro"))
    print("\n")
    print("weighted: ")
    print(recall_score(gold, predictions, average="weighted"))
    print("\n")

    with open("./src/video_node/performance.txt", "a") as f:
        f.write(
            str(top_n)
            + " campioni, inliers "
            + str(inliers_min)
            + ": \nmicro precision "
            + str(precision_score(gold, predictions, average="micro"))
            + " macro precision "
            + str(precision_score(gold, predictions, average="macro"))
            + "\nmicro recall "
            + str(recall_score(gold, predictions, average="micro"))
            + " macro recall "
            + str(recall_score(gold, predictions, average="macro"))
            + "\n\n"
        )

    return


def init(nn_matches, img1, img2):
    global HIGHER_INLIERS
    global BEST_HOMOGRAPHY
    global RESULT_IMAGE
    global GOOD_MATCHES

    HIGHER_INLIERS = 0
    # [ratio test filtering]
    matched1 = []
    matched2 = []
    good = []
    nn_match_ratio = 0.8  # Nearest neighbor matching ratio
    for m, n in nn_matches:
        if m.distance < nn_match_ratio * n.distance:
            matched1.append(kpts1[m.queryIdx])
            matched2.append(kpts2[m.trainIdx])
            good.append(m)
            # print(good)
    # [ratio test filtering]

    # choosing 4 random points
    points = []
    for i in range(4):
        p = random.choice(good)
        points.append(p)

    # extract the matched keypoints
    src_pts = np.float32([kpts1[m.queryIdx].pt for m in points]).reshape(-1, 1, 2)
    dst_pts = np.float32([kpts2[m.trainIdx].pt for m in points]).reshape(-1, 1, 2)

    # find homography matrix and do perspective transform
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    # print("Homography\n")
    # print(H)

    # [homography check]
    inliers1 = []
    inliers2 = []
    good_matches = []
    inlier_threshold = (
        2.5  # Distance threshold to identify inliers with homography check
    )
    for i, m in enumerate(matched1):
        col = np.ones((3, 1), dtype=np.float64)
        col[0:2, 0] = m.pt

        col = np.dot(H, col)
        col /= col[2, 0]
        dist = sqrt(
            pow(col[0, 0] - matched2[i].pt[0], 2)
            + pow(col[1, 0] - matched2[i].pt[1], 2)
        )

        if dist < inlier_threshold:
            good_matches.append(cv2.DMatch(len(inliers1), len(inliers2), 0))
            inliers1.append(matched1[i])
            inliers2.append(matched2[i])
            # print(good_matches)

    source = np.float32([kpts1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    destin = np.float32([kpts2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    # [homography check]

    # [draw final matches]
    res = np.empty(
        (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3),
        dtype=np.uint8,
    )
    cv2.drawMatches(img1, inliers1, img2, inliers2, good_matches, res)
    # cv2.imshow("img", res)
    # cv2.imwrite("matching_result.png", res)

    inlier_ratio = len(inliers1) / float(len(matched1))

    if len(inliers1) > HIGHER_INLIERS:
        HIGHER_INLIERS = len(inliers1)
        BEST_HOMOGRAPHY = H
        RESULT_IMAGE = res
        GOOD_MATCHES = good_matches

    return HIGHER_INLIERS, BEST_HOMOGRAPHY, RESULT_IMAGE, GOOD_MATCHES


def cos_similarity(v1, v2):
    return cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))


def retrieve_similar_image(img, top_n):
    # retrieve the vocabulary for the image in analysis
    vocab = image_voc[img][0]
    img_to_check = []

    for ix, v in tqdm(enumerate(vocab)):
        if int(v) > 0:
            img_to_check.append(table[str(ix)])

    # summing all the lists associated to different inverted_indeces
    img_to_check = np.unique(reduce(operator.add, img_to_check))
    similarities = defaultdict()

    # For each image that is relevant, compute the similarity score
    for image in tqdm(img_to_check):
        if image in frames:
            similarities[image] = cos_similarity(
                np.array(computed_tfidf[img]),  # image we choose
                np.array(computed_tfidf[image]),
            )[0][0]

    # Return the top n similar images
    out = dict(
        sorted(
            similarities.items(),
            key=lambda x: x[1],  # Choosing the second item from x
            reverse=True,
        )[:top_n]
    ).keys()
    scores = dict(
        sorted(
            similarities.items(),
            key=lambda x: x[1],  # Choosing the second item from x
            reverse=True,
        )[:top_n]
    ).values()

    folder = Path(TEST_IMAGES)
    image_list = [str(name).split("/")[-1] for name in folder.glob("*.jpg")]

    return out


if __name__ == "__main__":
    path = os.getcwd()
    bovw_path = path + "/src/pretrained/"
    img_path = path + "/src/video_node/images/"

    with open(os.path.join(bovw_path, "table.txt")) as f:
        table = json.loads(f.readlines()[0])

    with open(path + "/src/video_node/frames.txt", "r") as f:
        frames = json.loads(f.readlines()[0])

    with open(os.path.join(bovw_path, "tfidf.txt")) as f:
        computed_tfidf = f.readlines()
        computed_tfidf = json.loads(computed_tfidf[0])[0]

    with open(os.path.join(bovw_path, "image_vocabs.txt")) as f:
        image_voc = f.readlines()
        image_voc = json.loads(image_voc[0])[0]

    matches = []

    # for frame in tqdm(frames):
    immagine = "frame100335.jpg"
    out = list(retrieve_similar_image(immagine, top_n))
    print(out)

    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    img1 = cv2.imread(cv2.samples.findFile(img_path + immagine), cv2.IMREAD_GRAYSCALE)
    extractor = cv2.AKAZE_create()
    kpts1, desc1 = extractor.detectAndCompute(img1, None)

    for elem in tqdm(out):
        # print(elem)

        img2 = cv2.imread(cv2.samples.findFile(img_path + elem), cv2.IMREAD_GRAYSCALE)

        kpts2, desc2 = extractor.detectAndCompute(img2, None)

        nn_matches = matcher.knnMatch(desc1, desc2, 2)

        for i in range(100):
            inliers, homography, res, good_matches = init(nn_matches, img1, img2)

        """ print("\nInliers\n")
        print(inliers) """

        """  if inliers > inliers_min:
            matches.append(elem)

            # extract the matched keypoints
            src_pts = np.float32([kpts1[m.queryIdx].pt for m in good_matches]).reshape(
                -1, 1, 2
            )
            dst_pts = np.float32([kpts2[m.trainIdx].pt for m in good_matches]).reshape(
                -1, 1, 2
            )

            # find homography matrix considering all inliers
            M, _ = cv2.findHomography(src_pts, dst_pts, 0)

            # cv2.imshow("Result", res)
            cv2.imwrite("src/video_node/matches/matching_result" + elem + ".png", res)
            cv2.waitKey()
 """
    ris = ""
    perf = []
    for frame in frames:
        if frame in out:
            ris += "1\n"
            perf.append("1")
        else:
            ris += "0\n"
            perf.append("0")

    with open("./src/video_node/matches.txt", "w") as f:
        f.write(ris)

    performance(perf)
