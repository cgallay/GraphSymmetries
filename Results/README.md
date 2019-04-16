# Results

## ConvNet on Graph AID

commit: [4f7d65d](https://github.com/cgallay/Semester_Project/tree/4f7d65d956510ec0f16691bcfc926115505ad055)

cmd: `python3.6 main.py --arch ConvNet --lr=0.01 --wd=5e-4 --nb_epochs=300 --on_graph --dataset=AID`

Accuracy: 82.30%

Train set 0.8 and Test set 0.2

nb_param: 987'140

Speed: 18 min/epoch

## ConvNet on Graph AID

commit: [4f7d65d](https://github.com/cgallay/Semester_Project/tree/4f7d65d956510ec0f16691bcfc926115505ad055)

cmd: `python3.6 main.py --arch ConvNet --lr=0.01 --wd=5e-4 --nb_epochs=300 --dataset=AID`

Accuracy: 80.20%

Train set 0.8 and Test set 0.2

nb_param: 1'150'852

Speed: 18 min/epoch

## ConvNet on Graph

commit: [5910a43](https://github.com/cgallay/Semester_Project/tree/5910a43f38024c23ed2158d7e502ff2cf792f7ba)

cmd: `python3.6 main.py --arch ConvNet --on_graph --lr=0.005 --ls 75 100 125 150 --wd=5e-4 --nb_epochs=160`

Accuracy: > 70%


nb_para: 938'732


Speed: 3.5625 minutes / epoch


Note: the network was train without data augmentation and a kernel size of 5.

## ConvNet on Graph:
commit: [8cdccc2](https://github.com/cgallay/Semester_Project/tree/8cdccc2ebc6fd6ca73a27efef03bf78c156bceff)

cmd: `python3.6 main.py --arch ConvNet --lr=0.01 --wd=5e-4 --da --nb_epochs=160 --ls 75 125 150 --on_graph`

Accuracy: >73%

Speed: 4.125 minutes / epoch

nb_param: 1'551'116

## ConvNet - 20.03.2019
commit: [27780a2](https://github.com/cgallay/Semester_Project/tree/27780a226bb2f86129502d5bec1981e37cd34f4c)

cmd: `./run_best.sh ConvNet`

Accuracy: >88% at best

Speed: 1.55 minutes / epoch

nb_param: 2'296'556


## ConvNet on_graph - 17.03.2019
commit: [1649219](https://github.com/cgallay/Semester_Project/tree/164921931c932a8a3b77bc6f475be7e94bdb4729)

cmd: `python3.6 main.py --arch ConvNet --lr=0.02 --da --nb_epochs=300 --on_graph --ls 50 100 150 200 250 300`

**Note:** The comment has been run two time with [a change](https://github.com/cgallay/Semester_Project/blob/164921931c932a8a3b77bc6f475be7e94bdb4729/models/classics.py#L55) in the kernel_size once noraml (by commenting the line) and once multiply by two.
What have been observed is that the network doesn't totaly overfit in both case but accaracy is getting better with a larger kernel_size. 

Accuracy: 
- Test: 62% & 75% (with double `kernel_size`)
- Train: 77% & 93%

## ResNet - 06.03.2019
commit: [243c8ab](https://github.com/cgallay/Semester_Project/tree/243c8abe6d64a02e97a1c58be205843241444cb6)

cmd: `./run_best.sh ResNet18`

Accuracy: >94% at best

Speed: 3.5474min per epoch