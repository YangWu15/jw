diffenvrnn1.py - for training RNN and saving the trained model

decoderC.py - import the trained RNN to generate training sample to train the decoders, and save the decoders 

decoder_livetest.py - import the trained decoders and the trained RNN to generate data (in the length which is the decoder is trained on) live, in order to test the decoders

decoder_generalizationtest.py - import the trained decoders and the trained RNN to generate data of different lengths (decoder has not seen some lengths) live, in order to test the decoders

encoderC2.py - import the trained RNN to generate training sample to train the encoder, and use the encoder to do the manipulation test, then import the trained decoder and do the swap test (note: swap test is still constructing, there're SHAM SWAP (sham swap where we just use the same obs_x that we used to generate pattern_to_remove to feed into the same encoder to generate pattern_to_add, this is for diagnosing what could went wrong for this process) and NOT SHAM SWAP (regular swap where we actually look for a obs_y that is not in the specific sequence we're manipulating and feed it into the trained encoder to get a pattern_to_add)
