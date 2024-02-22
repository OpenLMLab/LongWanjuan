# DMR Score
We use relationships (determined by [DMR](https://arxiv.org/abs/2306.10658)) between adjacent sentences to measure the cohesion of long text. 
## Train Chinese DMR Model
The DMR approach is originally considered for English texts only. 
To process Chinese data, we follow its training methodology and train a Chinese DMR model based on the Wanjuan dataset.

We first get data from unsupervised corpus.
```bash
srun --ntasks-per-node=64 --ntasks=64 --cpus-per-task=1 python -u get_data.py
srun --ntasks-per-node=64 --ntasks=64 --cpus-per-task=1 python -u get_data.py
```

Then we train the model with data above.
```bash
python dmv_train.py
```

## Inference
To get the DMR score of a long text, we use the trained model to process the text and get the score.
```bash
srun --ntasks=128 --ntasks-per-node=8 --gres=gpu:8 python -u get_dmv_score.py
```
