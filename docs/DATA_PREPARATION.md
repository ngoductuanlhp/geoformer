## ScanNet v2 dataset

1\) Download the [ScanNet](http://www.scan-net.org/) v2 dataset.

2\) Extract the downloaded zip file to `geoformer/data` as follows.

```
geoformer
├── data
│   ├── scannetv2
│   │   ├── geo
│   │   ├── scenes
│   │   ├── support_sets
│   │   ├── class2instances.pkl
│   │   ├── class2scans.pkl
...
```
where `geo` is used for computing geodesic distance, `scenes` is the preprocessed 3d point cloud data stored in `.npy` format, `support_sets` stores the predefined support sets (used to eval all experiments)
