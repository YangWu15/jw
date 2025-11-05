diffenvrnn1.py - for training RNN and saving the trained model
decoderC.py - import the trained RNN to generate training sample to train the decoders, and save the decoders 
decoder_livetest.py - import the trained decoders and the trained RNN to generate data (in the length which is the decoder is trained on) live, in order to test the decoders
decoder_generalizationtest.py - import the trained decoders and the trained RNN to generate data of different lengths (decoder has not seen some lengths) live, in order to test the decoders
encoderC2.py - import the trained RNN to generate training sample to train the encoder, and use the encoder to do the manipulation test, then import the trained decoder and do the swap test
