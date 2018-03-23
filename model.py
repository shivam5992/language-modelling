from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku 
import numpy as np 

tokenizer = Tokenizer()

def dataset_preparation(data):

	# basic cleanup
	corpus = data.lower().split("\n")

	# tokenization	
	tokenizer.fit_on_texts(corpus)
	total_words = len(tokenizer.word_index) + 1

	# create input sequences using list of tokens
	input_sequences = []
	for line in corpus:
		token_list = tokenizer.texts_to_sequences([line])[0]
		for i in range(1, len(token_list)):
			n_gram_sequence = token_list[:i+1]
			input_sequences.append(n_gram_sequence)

	# pad sequences 
	max_sequence_len = max([len(x) for x in input_sequences])
	input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

	# create predictors and label
	predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
	label = ku.to_categorical(label, num_classes=total_words)

	return predictors, label, max_sequence_len, total_words

def create_model(predictors, label, max_sequence_len, total_words):
	
	model = Sequential()
	model.add(Embedding(total_words, 10, input_length=max_sequence_len-1))
	model.add(LSTM(150))
	model.add(Dense(total_words, activation='softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
	model.fit(predictors, label, epochs=100, verbose=1, callbacks=[earlystop])
	print model.summary()
	return model 