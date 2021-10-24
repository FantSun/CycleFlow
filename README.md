# CYCLEFLOW: PURIFY INFORMATION FACTORS BY CYCLE LOSS

This code is a pytorch version for CycleFlow model in "CycleFlow: Purify Information Factors by Cycle Loss", which is modified for from the original [Speechflow](https://github.com/auspicious3000/SpeechSplit).

Some toolkits to be used can be found at our modified [SpeechFlow](https://github.com/FantSun/Speechflow)

## Dependencies

This project is built with python 3.6, for other packages, you can install them by ```pip install -r requirements.txt```.


## To Prepare Training Data(take VCTK as an example here, for other dataset, you should modify some settings in below files)

1. Prepare your wavefiles

2. Prepare spectrograms and pitch contours for your data following tools provided in [SpeechFlow](https://github.com/FantSun/Speechflow)


## To Train

1. change settings at ```hparams.py``` and ```run.py```

2. Run the training scripts: ```python run.py```


## To implement inference
You can change inputs of the "Generator_loop" module to make the model adapted for your inference.


## Final Words

This is a code for "CycleFlow: Purify Information Factors by Cycle Loss"([demo](http://cycleflow.cslt.org/), [paper](http://cycleflow.cslt.org/paper.pdf)), this improved model can produce more disentangled factors than SpeechFlow. The main design in this model is a random factor substitution (RFS) operation and a cycle loss. We highlight that this technique is simple and general and can be applied to any factorization model, although in this study we test it with the SpeechFlow architecture, hence the name CycleFlow.
This code is modified for our task from the original [Speechflow](https://github.com/auspicious3000/SpeechSplit). Thanks for Kaizhi Qian providing the original code, which is much helpful for us.
