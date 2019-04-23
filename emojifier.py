import numpy as np
np.random.seed(0)
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
from emo_utils import * 
np.random.seed(1)

def sentences_to_indices(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()` (described in Figure 4). 
    
    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this. 
    
    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """
    m = X.shape[0]

    X_indices = np.zeros((m, max_len))
    for i in range(m):
    	sentence_words = X[i].lower().split()
    	j = 0
    	for w in sentence_words:
    		X_indices[i][j] = word_to_index[w]
    		j = j + 1

    return X_indices

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')

##X1 = np.array(["funny lol", "lets play baseball", "food is ready for you"])
##X1_indices = sentences_to_indices(X1, word_to_index, 5)
##print("X1 = ", X1)
##print("X1_indices = ", X1_indices )

def pretrained_embedding_layer(word_to_vec_map, word_to_index):
	vocab_len = len(word_to_index) + 1 # adding 1 to fit Keras embedings(requirement)
	emb_dim = word_to_vec_map["cucumber"].shape[0]

	# Initialize the embedding matrix as a numpy array of zeros of shape (vocab_len, dimensions of word vectors = emb_dim)
	emb_matrix = np.zeros((vocab_len, emb_dim))

	for word, index in word_to_index.items():
		emb_matrix[index, :] = word_to_vec_map[word]

	embedding_layer = Embedding(vocab_len, emb_dim, trainable = False)

	embedding_layer.build((None, ))

	embedding_layer.set_weights([emb_matrix])

	return embedding_layer

embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
#print(embedding_layer.get_weights()[0][1][3])

def Emojify_v2(input_shape, word_to_vec_map, word_to_index):
	sentence_indices = Input(shape = input_shape, dtype = 'int32')
	embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
	embeddings = embedding_layer(sentence_indices)
	X = LSTM(128, return_sequences = True)(embeddings)
	X = Dropout(0.5)(X)
	X = LSTM(128, return_sequences = False)(X)
	X = Dense(5)(X)
	X = Activation('softmax')(X)
	model = Model(inputs = sentence_indices, output = X)
	return model 

model = Emojify_v2((10, ), word_to_vec_map, word_to_index)
#model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

X_train, Y_train = read_csv('data/train_emoji.csv')
X_test, Y_test = read_csv('data/tesss.csv')

maxLen = 10

X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
Y_train_oh = convert_to_one_hot(Y_train, C = 5)

model.fit(X_train_indices, Y_train_oh, epochs = 50, batch_size = 32, shuffle = True)

X_test_indices = sentences_to_indices(X_test, word_to_index, max_len = maxLen)
Y_test_oh = convert_to_one_hot(Y_test, C = 5)
loss, acc = model.evaluate(X_test_indices, Y_test_oh)
print("Test accuracy = ", acc)

X_test_indices = sentences_to_indices(X_test, word_to_index, maxLen)
pred = model.predict(X_test_indices)
for i in range(len(X_test)):
    x = X_test_indices
    num = np.argmax(pred[i])
    if(num != Y_test[i]):
        print('Expected emoji:'+ label_to_emoji(Y_test[i]) + ' prediction: '+ X_test[i] + label_to_emoji(num).strip())