### Data preparation

We used the coreference resolution model from [Zhang et al., 2023](https://aclanthology.org/2023.emnlp-main.704.pdf) to extract coreferences. You can download the transcripts and extracted coreferences from [here](https://drive.google.com/drive/folders/1IMEGyoe5U0zOH83daqncfN0iSbdoxDhw?usp=drive_link). 

After downloading the data, you can run `extract_conven.py` and `prepare_dataset.py` to process them. 

### Training

For SFT training, run `sft_training.sh`. For preference optimization training, run `po_training.sh` . 


