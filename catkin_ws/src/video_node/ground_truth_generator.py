#! /usr/bin/env python
import json

if __name__ == "__main__":
    frames = []
    gold = ""
    for i in range(100000, 102559):
        frames.append("frame" + str(i) + ".jpg")
        if (
            (i <= 100355 and i >= 100335)
            or (i <= 100585 and i >= 100565)
            or (i <= 101255 and i >= 101235)
            or (i <= 101445 and i >= 101425)
            or (i <= 102170 and i >= 102150)
            or (i <= 102170 and i >= 102150)
        ):
            gold += "1 \n"
        else:
            gold += "0 \n"

    with open("src/video_node/ground_truth.txt", "w") as f:
        f.write(gold)

    with open("src/video_node/frames.txt", "w") as f:
        json.dump(frames, f)
