# rnn_trainer.py
# Code that trains on the Screenplay data and produces a model for text generation.

import numpy as np
import operator
import theano as theano
import theano.tensor as T
import time
import sys

from Preprocess import ScreenplayData
from utils import *

class RNN_Theano_Trainer:
	# This class has been written based on the tutorial detailed here:
	# http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        # Theano: Created shared variables
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))      
        # We store the Theano graph here
        self.theano = {}
        self.__theano_build__()
    
    def __theano_build__(self):
        U, V, W = self.U, self.V, self.W
        x = T.ivector('x')
        y = T.ivector('y')
        def forward_prop_step(x_t, s_t_prev, U, V, W):
            s_t = T.tanh(U[:,x_t] + W.dot(s_t_prev))
            o_t = T.nnet.softmax(V.dot(s_t))
            return [o_t[0], s_t]
        [o,s], updates = theano.scan(
            forward_prop_step,
            sequences=x,
            outputs_info=[None, dict(initial=T.zeros(self.hidden_dim))],
            non_sequences=[U, V, W],
            truncate_gradient=self.bptt_truncate,
            strict=True)
        
        prediction = T.argmax(o, axis=1)
        o_error = T.sum(T.nnet.categorical_crossentropy(o, y))
        
        # Gradients
        dU = T.grad(o_error, U)
        dV = T.grad(o_error, V)
        dW = T.grad(o_error, W)
        
        # Assign functions
        self.forward_propagation = theano.function([x], o)
        self.predict = theano.function([x], prediction)
        self.ce_error = theano.function([x, y], o_error)
        self.bptt = theano.function([x, y], [dU, dV, dW])
        
        # SGD
        learning_rate = T.scalar('learning_rate')
        self.sgd_step = theano.function([x,y,learning_rate], [], 
                      updates=[(self.U, self.U - learning_rate * dU),
                              (self.V, self.V - learning_rate * dV),
                              (self.W, self.W - learning_rate * dW)])
    
    def calculate_total_loss(self, X, Y):
        return np.sum([self.ce_error(x,y) for x,y in zip(X,Y)])
    
    def calculate_loss(self, X, Y):
        # Divide calculate_loss by the number of words
        num_words = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X,Y)/float(num_words)

def gradient_check_theano(model, x, y, h=0.001, error_threshold=0.01):
    # Overwrite the bptt attribute. We need to backpropagate all the way to get the correct gradient
    model.bptt_truncate = 1000
    # Calculate the gradients using backprop
    bptt_gradients = model.bptt(x, y)
    # List of all parameters we want to chec.
    model_parameters = ['U', 'V', 'W']
    # Gradient check for each parameter
    for pidx, pname in enumerate(model_parameters):
        # Get the actual parameter value from the mode, e.g. model.W
        parameter_T = operator.attrgetter(pname)(model)
        parameter = parameter_T.get_value()
        print("Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape)))
        # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
        it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            # Save the original value so we can reset it later
            original_value = parameter[ix]
            # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
            parameter[ix] = original_value + h
            parameter_T.set_value(parameter)
            gradplus = model.calculate_total_loss([x],[y])
            parameter[ix] = original_value - h
            parameter_T.set_value(parameter)
            gradminus = model.calculate_total_loss([x],[y])
            estimated_gradient = (gradplus - gradminus)/(2*h)
            parameter[ix] = original_value
            parameter_T.set_value(parameter)
            # The gradient for this parameter calculated using backpropagation
            backprop_gradient = bptt_gradients[pidx][ix]
            # calculate The relative error: (|x - y|/(|x| + |y|))
            relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
            # If the error is to large fail the gradient check
            if relative_error > error_threshold:
                print("Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix))
                print("+h Loss: %f" % gradplus)
                print("-h Loss: %f" % gradminus)
                print("Estimated_gradient: %f" % estimated_gradient)
                print("Backpropagation gradient: %f" % backprop_gradient)
                print("Relative Error: %f" % relative_error)
                return 
            it.iternext()
        print("Gradient check for parameter %s passed." % (pname))

class RNN_Numpy_Trainer:
	# This class has been written based on the tutorial detailed here:
	# http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/

	def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
		# Assign instance variables
		self.word_dim = word_dim
		self.hidden_dim = hidden_dim
		self.bptt_truncate = bptt_truncate
		
		# Randomly initialize the network parameters
		self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
		self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
		self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))

	@staticmethod
	def softmax(w, t = 1.0):
		e = np.exp(np.array(w) / t)
		return e / np.sum(e)

	def forward_propagation(self, x):
		# The total number of time steps
		T = len(x)

		# During forward propagation we save all hidden states in s because need them later.
		# We add one additional element for the initial hidden, which we set to 0
		s = np.zeros((T + 1, self.hidden_dim))
		s[-1] = np.zeros(self.hidden_dim)
		# The outputs at each time step. Again, we save them for later.
		o = np.zeros((T, self.word_dim))
		# For each time step...
		for t in np.arange(T):
			# Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
			s[t] = np.tanh(self.U[:,x[t]] + self.W.dot(s[t-1]))
			o[t] = RNN_Numpy_Trainer.softmax(self.V.dot(s[t]))
			return [o, s]

	def predict(self, x):
		# Perform forward propagation and return index of the highest score
		o, s = self.forward_propagation(x)
		return np.argmax(o, axis=1)

	def calculate_total_loss(self, x, y):
	    L = 0
	    # For each sentence...
	    for i in np.arange(len(y)):
	        o, s = self.forward_propagation(x[i])
	        # We only care about our prediction of the "correct" words
	        correct_word_predictions = o[np.arange(len(y[i])), y[i]]
	        # Add to the loss based on how off we were
	        L += -1 * np.sum(np.log(correct_word_predictions))
	    return L

	def calculate_loss(self, x, y):
	    # Divide the total loss by the number of training examples
	    N = np.sum((len(y_i) for y_i in y))
	    return self.calculate_total_loss(x,y)/N

class Generate_Text:
    def __init__(self):
    	pass

    def generate_sentence(self, model, word_to_index, index_to_word):
    	unknown_token = "UNKNOWN_TOKEN"
    	sentence_start_token = "SENTENCE_START"
    	sentence_end_token = "SENTENCE_END"

    	# We start the sentence with the start token
    	new_sentence = [word_to_index[sentence_start_token]]

    	# Repeat until we get an end token
    	while not new_sentence[-1] == word_to_index[sentence_end_token] and len(new_sentence) < 25:
    		next_word_probs = model.forward_propagation(new_sentence)
    		sampled_word = word_to_index[unknown_token]
    		# We don't want to sample unknown words
    		while sampled_word == word_to_index[unknown_token]:
    			samples = np.random.multinomial(1, next_word_probs[-1])
    			sampled_word = np.argmax(samples)
    		new_sentence.append(sampled_word)
    	sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
    	return sentence_str

if __name__ == "__main__":
	vocab_size = sys.argv[1] if len(sys.argv) > 1 else 20000
	train = ScreenplayData("../data/processed/lines_by_genre.db", vocab_size)

	model = RNN_Theano_Trainer(vocab_size, hidden_dim=400)
	t1 = time.time()
	model.sgd_step(train.X[10], train.Y[10], 0.005)
	t2 = time.time()
	print("SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.))

	print("Generating sentences...")

	generate_text_model = Generate_Text()
	num_sentences = 50
	sentence_min_length = 7

	for i in range(num_sentences):
		sent = []
		# We want long sentences, not sentences with one or two words
		while len(sent) < sentence_min_length:
			sent = generate_text_model.generate_sentence(model, train.word_to_index, train.index_to_word)
		print(" ".join(sent) + "\n\n")

	# np.random.seed(10)
	# model = RNN_Numpy_Trainer(vocab_size)
	# o, s = model.forward_propagation(train.X[10])
	# print(o.shape)
	# print(o)

	# predictions = model.predict(train.X[10])
	# print(predictions.shape)
	# print(predictions)



