##VUI python file 

###DATA AND CONCEPT EXPLORATION
## The pipeline takes in raw audio input and gives a text output.
## Goal :- 1- Add a language model to improve this translation
##		   2- Summarization and Analysis 

##LibriSpeech dataset - 1000 hours of speech derived from audiobooks
##					  - used to train and evaluate asr models

from data_generator import vis_train_features

vis_text, vis_raw_audio, vis_mfcc_feature, vis_spectrogram_feature, vis_audio_path = vis_train_features()

