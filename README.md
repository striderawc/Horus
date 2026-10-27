# Horus

Efficient Real Time Tennis Ball Tracking and Near Real Time Shot Analysis

## Dataset

Uses the [TrackNet](https://nol.cs.nctu.edu.tw/ythuang/epapers/ICIP18_TrackNet.pdf) tennis dataset (place under `Dataset/`, gitignored):

```text
Dataset/
  game1/Clip1/0000.jpg … Label.csv
  …
  game10/…
```

Frames are 1280×720 @ 30 fps. Each `Label.csv` has `file name`, `visibility`, `x-coordinate`, `y-coordinate`, `status`.

Default splits (by game, no clip leakage):

| Split | Games |
|-------|-------|
| train | game1–game7 |
| val   | game8–game9 |
| test  | game10 |

