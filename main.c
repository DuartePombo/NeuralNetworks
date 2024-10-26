#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// parameters for the MNIST dataset: Input size = 28x28 pixels, with an hidden layer of 128 neurons, output = 10, to classify numbers between 0 and 9.
#define INPUT_SIZE 784 
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01


double weights_input_hidden[INPUT_SIZE][HIDDEN_SIZE];
double weights_hidden_output[HIDDEN_SIZE][OUTPUT_SIZE];
double bias_hidden[HIDDEN_SIZE];
double bias_output[OUTPUT_SIZE];


void initialize_weights(){

  // initialize Input and hidden layer weights to random values between -1 and 1.
  for(int i=0; i<INPUT_SIZE; i++){
    for(int j=0; j<HIDDEN_SIZE; j++){
      weights_input_hidden[i][j] = = ((double)rand()/RAND_MAX)*2.0 - 1.0;
    }
  }

  // initialize hidden layer and output layer weights
  for(int i=0; i<HIDDEN_SIZE; i++){
    for(int j=0; j<OUTPUT_SIZE; j++){
      weights_hidden_output[i][j] = ((double)rand()/RAND_MAX)*2.0 - 1.0;

    }
  }

  for(int i=0; i<HIDDEN_SIZE; i++){
    bias_hidden[i] = ((double)rand()/RAND_MAX) * 2.0 - 1.0;

  }

  for(int i=0; i<OUTPUT_SIZE; i++){
    bias_output[i] = ((double)rand()/RAND_MAX) * 2.0 - 1.0;
  }

}

// sigmoid activation function
double sigmoid(double z){
  return 1.0/ (1.0 + exp(-z));
}

double sigmoid_derivative(double sig){
  return  sig * (1.0 - sig); // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
}

// ReLu activation 
double relu(double x){
  return x > 0? x:0;
}

double relu_derivative(double x){
  return z>0? 1:0;
}

// transforms the logits in the final output layer, to the softmax probability values.
void softmax(double *output, int size){
  double sum = 0.0;

  for (int logit = 0; logit < size; logit++){
    output[logit] = exp(output[logit]); //e^z
    sum += output[logit]; // sum(e^z)
  }

  for(int logit = 0; logit< size; logit++){
    output[logit] /= sum; // output becomes the softmax probability
  }

}

void forward_pass(double *input, double *hidden_output, double *final_output) {

  //hidden layer calculations
  for (int i=0; i<HIDDEN_SIZE; i++){
    
    hidden_output[i] = 0.0; //initialize output at 0

    for (int j=0; j<INPUT_SIZE; j++){
      hidden_output[i] = hidden_output[i] + weights_input_hidden[j][i]*input[j]; //input weight for neuron j connected to hidden neuron i, this is hidden output before activation function, and it is the summation of all the weights connected to that particular neuron from all the inputs.
    }
    hidden_output[i] = hidden_output[i] + bias_hidden[i]; // add the constant term
    hidden_output[i] = relu(hidden_output[i]); // pass it through activation function
  }

  // Outpuyt layer:

  for(int i=0; i<OUTPUT_SIZE;i++){
    final_output[i]+=0.0 //initalize output at 0
    for(int j=0; j<HIDDEN_SIZE;j++){
        
      final_output[i] = final_output[i] + hidden_output[j]*weights_hidden_output[j][i];
    }
    final_output[i]+=bias_output[i];
  }

  softmax(final_output, OUTPUT_SIZE);

}
