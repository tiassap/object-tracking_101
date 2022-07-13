# object-tracking_101

Basic implementation of object tracking algorithm.


[![Watch the video](https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2FNNet%2F2BnG8XEZoH.png?alt=media&token=25e2386e-4ea5-499a-8981-117a1e5941d8)](https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2FNNet%2FrOVg603WnZ.mp4?alt=media&token=dbf5cea6-d982-4758-99a4-0c36ddadc253)
_Object tracking with SORT algorithm on MOT-15. Faster R-CNN is used as object detection algorithm. Still have some problems with occlusion here. (Short term occlusion is okay)._

Run this command line for real-time inference using video camera. Note: press `Esc` to exit.
```
$ python main.py 
```

For inference on video:
```
$ python main.py --video <path/to/video>
```

To save output video:
```
$ python main.py --save
```

### 1. SORT (Simple Online and Realtime Tracking)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12l5QsXrjj0l0PlkYUp3lpjHjrlaBRXUV#scrollTo=X_Qg774xJoMr&uniqifier=2)
```
@inproceedings{Bewley2016_sort,
  author={Bewley, Alex and Ge, Zongyuan and Ott, Lionel and Ramos, Fabio and Upcroft, Ben},
  booktitle={2016 IEEE International Conference on Image Processing (ICIP)},
  title={Simple online and realtime tracking},
  year={2016},
  pages={3464-3468},
  keywords={Benchmark testing;Complexity theory;Detectors;Kalman filters;Target tracking;Visualization;Computer Vision;Data Association;Detection;Multiple Object Tracking},
  doi={10.1109/ICIP.2016.7533003}
}
```

