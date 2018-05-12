# pytorch_qanet
pytorch implementation of QANet for the squad dataset challenge

To load previous model:
You can change experiment number in config.py and then choose which epoch you want to load from. 
The best result I got (F1: 62.970) is experiment_9/model_checkpoints/checkpoint_epoch_4. It is the only model I included for this submission. I can give you more if you need them though. The F1 and loss graphs are also in the experiment_9 folder. 

You will also need to change some path names that are hard coded in the config.py file. 

Data:
This zip file contains the glove embedding files under data_files/glove_txt
The squad datasets are stored under data_files/squad_datasets

