## cwFedAvg for Personalized Federated Learning

### Dataset setting
```sh
cd dataset
python generate_cifar10.py noniid - dir
```

### PFL training using cwFedAvg algorithm
```sh
# cwFedAvg with WDR
cd system
python main.py -lbs 10 -nc 20 -jr 1 -nb 10 -data Cifar10 -m cnn -algo cwFedAvg -gr 1000 -cw -wdr -wd 10 -did 0 -go cnn

# cwFedAvg without WDR
cd system
python main.py -lbs 10 -nc 20 -jr 1 -nb 10 -data Cifar10 -m cnn -algo cwFedAvg -gr 1000 -cw -did 0 -go cnn

# cwFedAvg with Local Class Distribution
cd system
python main.py -lbs 10 -nc 20 -jr 1 -nb 10 -data Cifar10 -m cnn -algo cwFedAvg -gr 1000 -cw -gt -did 0 -go cnn

```

The code for the experiment was developed based on 
Zhang, J.; Liu, Y.; Hua, Y.; Wang, H.; Song, T.; Xue, Z.; Ma, R.; and Cao, J. 2023b. Pfllib: Personalized federated learning algorithm library. arXiv preprint arXiv:2312.04992.
