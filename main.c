#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mnist.h"
// parameters for the MNIST dataset: Input size = 28x28 pixels, with an hidden layer of 128 neurons, output = 10, to classify numbers between 0 and 9.
#define INPUT_SIZE 784 // 28*28 = 784 
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01


double weights_input_hidden[INPUT_SIZE][HIDDEN_SIZE];
double weights_hidden_output[HIDDEN_SIZE][OUTPUT_SIZE];
double bias_hidden[HIDDEN_SIZE];
double bias_output[OUTPUT_SIZE];



// Function to print a progress bar during training
void print_progress_bar(int current, int total) {
    int bar_width = 50;
    float progress = (float)current / total;
    
    printf("[");
    int pos = bar_width * progress;
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) printf("=");
        else if (i == pos) printf(">");
        else printf(" ");
    }
    printf("] %d%%\r", (int)(progress * 100));
    fflush(stdout); // Force the output to be displayed
}




void initialize_weights(){
  printf("Initializing weights...\n");
  // initialize Input and hidden layer weights to random values between -1 and 1.
  for(int i=0; i<INPUT_SIZE; i++){
    for(int j=0; j<HIDDEN_SIZE; j++){
      weights_input_hidden[i][j] = ((double)rand()/RAND_MAX)*2.0 - 1.0;
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
  printf("Weights initialization complete.\n\n");
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
  return x>0? 1:0;
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
    final_output[i] = 0.0; //initalize output at 0
    for(int j=0; j<HIDDEN_SIZE;j++){
        
      final_output[i] = final_output[i] + hidden_output[j]*weights_hidden_output[j][i];
    }
    final_output[i]+=bias_output[i];
  }

  softmax(final_output, OUTPUT_SIZE);

}

void backward_pass(double *input, double *hidden_output, double *final_output, double *target){

  double output_error[OUTPUT_SIZE];
  double hidden_error[HIDDEN_SIZE];

  // get error
  for(int i=0; i<OUTPUT_SIZE; i++){
  
  // We are using MSE as the loss function. The gradient (or derivative) of MSE with respect to the output
  // is simply (output - target). This gradient indicates the direction in which the weights should be adjusted to reduce the loss.
  // In backpropagation, we want to minimize the loss function by adjusting the weights of the neural network. To do this, we need to calculate
  // how much ecah output neuron contributes to the overal error. This contribution is what we call here by the output error. 
  // The outpout error tells us how far the model's predictions are from the true values and we further use this to adjust the weights.

    output_error[i] = final_output[i] - target[i]; 
  }

  // Backpropagation to the hidden layer:
  // We are going to compute how much each hidden neuron contributed to the overall error in the output layer.
  for(int i=0; i<HIDDEN_SIZE; i++){
    hidden_error[i] = 0.0;
    for(int j=0; j<OUTPUT_SIZE; j++){
      hidden_error[i] += output_error[j]*weights_hidden_output[i][j]; //sum the contribution of each output error. The error contribution of each output neuron j is scaled by the corresponding weight. This gives the total error contribution of the hidden neuron i to all the output neurons.
    }
    hidden_error[i] = hidden_error[i]*relu_derivative(hidden_output[i]); //since relu derivative is 1 for x>0, 0 otherwise, if the neuron was active, the error will propagate back. Only the neurons that were active during the forward pass are updated during back prop.

  }
  

  // updae the weights and biases between hidden and output layer
  for (int i=0; i<HIDDEN_SIZE; i++){
    for(int j=0; j<OUTPUT_SIZE; j++){
      weights_hidden_output[i][j] = weights_hidden_output[i][j] - (LEARNING_RATE * output_error[j] * hidden_output[i]);
    }
  }

  for(int i = 0; i<OUTPUT_SIZE; i++){
    bias_output[i] = bias_output[i] - (LEARNING_RATE*output_error[i]);
  }


  // Finally update weights and biases between input and hidden layer
  for (int i = 0; i < INPUT_SIZE; i++) {
      for (int j = 0; j < HIDDEN_SIZE; j++) {
          weights_input_hidden[i][j] -= LEARNING_RATE * hidden_error[j] * input[i];
      }
  }
  for (int i = 0; i < HIDDEN_SIZE; i++) {
      bias_hidden[i] -= LEARNING_RATE * hidden_error[i];
  }

}


void train(double **inputs, double **targets, int num_samples, int epochs) {
    double hidden_output[HIDDEN_SIZE];
    double final_output[OUTPUT_SIZE];

    for (int epoch = 0; epoch < epochs; epoch++) {
        printf("Epoch %d/%d started...\n", epoch + 1, epochs);

        for (int i = 0; i < num_samples; i++) {
            // Update progress bar for this epoch
            print_progress_bar(i + 1, num_samples);

            forward_pass(inputs[i], hidden_output, final_output);
            backward_pass(inputs[i], hidden_output, final_output, targets[i]);
        }

        printf("\nEpoch %d complete\n\n", epoch + 1);
    }
}

int predict(double *input) {
    double hidden_output[HIDDEN_SIZE];
    double final_output[OUTPUT_SIZE];

    forward_pass(input, hidden_output, final_output);

    int predicted_label = 0;
    double max_prob = final_output[0];
    for (int i = 1; i < OUTPUT_SIZE; i++) {
        if (final_output[i] > max_prob) {
            max_prob = final_output[i];
            predicted_label = i;
        }
    }
    return predicted_label;
}


// The mnist labels are stored as integers 0 to 9, but we are expecting one hot enconding.
// This function converts the labels to one hot enconding.
void one_hot_encode(int label, double *output) {
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = (i == label) ? 1.0 : 0.0;
    }
}



int main() {
    // Load the MNIST dataset
    printf("Loading MNIST dataset...\n");
    load_mnist();
    printf("MNIST dataset loaded.\n\n");

    int epochs = 10;
    int num_samples = NUM_TRAIN;
    int num_test_samples = NUM_TEST; // For testing
    
    // Prepare pointers for the training data
    double *inputs_pointers[NUM_TRAIN];
    double *targets_pointers[NUM_TRAIN];
    
    // Temporary array for one-hot encoding
    double one_hot[OUTPUT_SIZE];


    printf("Preparing training data...\n");
    for (int i = 0; i < num_samples; i++) {
        inputs_pointers[i] = train_image[i];

        // One-hot encode the training labels
        one_hot_encode(train_label[i], one_hot);
        targets_pointers[i] = malloc(OUTPUT_SIZE * sizeof(double));
        memcpy(targets_pointers[i], one_hot, OUTPUT_SIZE * sizeof(double));
    }

    printf("Training data preparation complete.\n\n");
    // Initialize weights
    initialize_weights();

    // Train the neural network
    train(inputs_pointers, targets_pointers, num_samples, epochs);

    // Free allocated memory for one-hot labels
    for (int i = 0; i < num_samples; i++) {
        free(targets_pointers[i]);
    }

    // ---------------------- Test the Model ----------------------

    printf("Testing the model...\n\n");

    int correct_predictions = 0;

    for (int i = 0; i < num_test_samples; i++) {
        int predicted_label = predict(test_image[i]);
        if (predicted_label == test_label[i]) {
            correct_predictions++;
        }
    }

    // Calculate and print the accuracy
    double accuracy = (double)correct_predictions / num_test_samples * 100.0;
    printf("Test Accuracy: %.2f%%\n\n", accuracy);

    return 0;
}

