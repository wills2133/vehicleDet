# vehicleDet
a real-time vehicle detecting program

Hog + Adaboost
<br/>
<img src="https://github.com/wills2133/vehicleDet/blob/master/recognition.JPG" width="320" height="160">
<br/>

Multitrack
<br/>
<img src="https://github.com/wills2133/vehicleDet/blob/master/multi-tracking.JPG" width="320" height="160">
<br/>

### Prerequisites

OpenCV

### Installing

Makefile in Linux

### How to run

@your_linux_dir:~$ vehDet 'video dir' -v 'choose a model.xml'20160306.xml

```
~$ vehDet ../test_video.avi -v 'choose a model.xml'
```
#degub control

```
///////////////////////////////////
//
//      key function:
//
// o/p: om/off capture frame
// v: show original frame
// f: show current det result
// w s, d c, z x, e r: adjust margin
// u j h k: adjust roi positon
// y i: change sacel size
// b: process the same frame
// n: normally detect
// sapce: pause/next fame
// t: show track
// g: skip processing/do not skip
// 0-9: show different roi
// a: do NMS
// m: auto/manually move 
// q: quit
//
///////////////////////////////////

```

## Authors

**Wills Wu**

