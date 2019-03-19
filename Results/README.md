# Results

## ConvNet on_graph - 17.03.2019
commit: [1649219](https://github.com/cgallay/Semester_Project/commit/164921931c932a8a3b77bc6f475be7e94bdb4729)

cmd: `python3.6 main.py --arch ConvNet --lr=0.02 --da --nb_epochs=300 --on_graph --ls 50 100 150 200 250 300`

**Note:** The comment has been run two time with [a change](https://github.com/cgallay/Semester_Project/blob/164921931c932a8a3b77bc6f475be7e94bdb4729/models/classics.py#L55) in the kernel_size once noraml (by commenting the line) and once multiply by two.
What have been observed is that the network doesn't totaly overfit in both case but accaracy is getting better with a larger kernel_size. 

Accuracy: 
- Test: 62% & 75% (with double `kernel_size`)
- Train: 77% & 93%

## ResNet - 06.03.2019
commit: [243c8ab](https://github.com/cgallay/Semester_Project/commit/243c8abe6d64a02e97a1c58be205843241444cb6)

cmd: `./run_best.sh ResNet18`

Accuracy: >94% at best

Speed: 3.5474min per epoch