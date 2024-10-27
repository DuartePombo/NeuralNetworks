#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mnist.h"

#define INPUT_SIZE 784 // 28x28 = 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10

double weights_input_hidden[INPUT_SIZE][HIDDEN_SIZE];
double weights_hidden_output[HIDDEN_SIZE][OUTPUT_SIZE];
double bias_hidden[HIDDEN_SIZE];
double bias_output[OUTPUT_SIZE];

// Function to load the trained model (weights and biases) from a file
void load_model(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Could not open file for loading weights!\n");
        return;
    }
    fread(weights_input_hidden, sizeof(double), INPUT_SIZE * HIDDEN_SIZE, file);
    fread(weights_hidden_output, sizeof(double), HIDDEN_SIZE * OUTPUT_SIZE, file);
    fread(bias_hidden, sizeof(double), HIDDEN_SIZE, file);
    fread(bias_output, sizeof(double), OUTPUT_SIZE, file);
    fclose(file);
    printf("Model loaded from %s\n", filename);
}

// ReLu activation function
double relu(double x) {
    return x > 0 ? x : 0;
}

// Softmax function for output layer probabilities
void softmax(double *output, int size) {
    double sum = 0.0;
    for (int logit = 0; logit < size; logit++) {
        output[logit] = exp(output[logit]);
        sum += output[logit];
    }
    for (int logit = 0; logit < size; logit++) {
        output[logit] /= sum;
    }
}

// Forward pass for inference
void forward_pass(double *input, double *hidden_output, double *final_output) {
    // Hidden layer calculations
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden_output[i] = 0.0;
        for (int j = 0; j < INPUT_SIZE; j++) {
            hidden_output[i] += weights_input_hidden[j][i] * input[j];
        }
        hidden_output[i] += bias_hidden[i]; // Add the bias
        hidden_output[i] = relu(hidden_output[i]); // Apply ReLU activation
    }

    // Output layer calculations
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        final_output[i] = 0.0;
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            final_output[i] += hidden_output[j] * weights_hidden_output[j][i];
        }
        final_output[i] += bias_output[i]; // Add the bias
    }

    // Apply softmax to get probabilities
    softmax(final_output, OUTPUT_SIZE);
}

// Predict the label for a single input
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

int main() {
    // Load the MNIST test dataset
    printf("Loading MNIST dataset...\n");
    load_mnist();
    printf("MNIST dataset loaded.\n");

    // Load the pre-trained model from the binary file
    load_model("trained_model.bin");

    // Perform inference on the test set
    printf("Running inference on the test set...\n");

    int num_test_samples = NUM_TEST;
    int correct_predictions = 0;

    // Iterate over the test set and make predictions
    for (int i = 0; i < num_test_samples; i++) {
        int predicted_label = predict(test_image[i]);
        if (predicted_label == test_label[i]) {
            correct_predictions++;
        }
    }

    // Calculate and print the accuracy
    double accuracy = (double)correct_predictions / num_test_samples * 100.0;
    printf("Test Accuracy: %.2f%%\n", accuracy);

    return 0;
}

