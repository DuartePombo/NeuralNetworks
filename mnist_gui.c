#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>

// Define constants
#define INPUT_SIZE 784 // 28x28
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define GRID_SIZE 28
#define CELL_SIZE 20  // Size of each cell in pixels
#define WINDOW_WIDTH (GRID_SIZE * CELL_SIZE)
#define WINDOW_HEIGHT (GRID_SIZE * CELL_SIZE + 150) // Extra space for buttons/text

// Neural network variables
double weights_input_hidden[INPUT_SIZE][HIDDEN_SIZE];
double weights_hidden_output[HIDDEN_SIZE][OUTPUT_SIZE];
double bias_hidden[HIDDEN_SIZE];
double bias_output[OUTPUT_SIZE];

// Global variables
Uint8 grid[GRID_SIZE][GRID_SIZE];
Uint32 last_update[GRID_SIZE][GRID_SIZE]; // Tracks last update time for each cell
enum Tool { DRAW, ERASE } current_tool = DRAW;
int quit = 0;
int predicted_label = -1;
double confidence = 0.0;

// Mouse state variables
int is_drawing = 0;                           // Flag to indicate if drawing is active
int current_mouse_x = 0, current_mouse_y = 0; // Current mouse position
const Uint32 UPDATE_INTERVAL = 1;            // Milliseconds between increments
const Uint8 STEP_VALUE = 50;                   // Increment/Decrement step value

// Button structure
typedef struct {
    SDL_Rect rect;         // Position and size of the button
    SDL_Color color;       // Button color
    char label[32];        // Button label text
    void (*action)();      // Function to call when the button is clicked
} Button;

#define BUTTON_WIDTH 100
#define BUTTON_HEIGHT 40

Button predict_button;
Button clear_button;
Button exit_button;

// Function prototypes
void load_model(const char *filename);
double relu(double x);
void softmax(double *output, int size);
void forward_pass(double *input, double *hidden_output, double *final_output);
int predict(double *input, double *confidence);
void render_grid(SDL_Renderer *renderer, Uint8 grid[GRID_SIZE][GRID_SIZE]);
int predict_grid(Uint8 grid[GRID_SIZE][GRID_SIZE], double *confidence);
void init_buttons();
void render_button(SDL_Renderer *renderer, TTF_Font *font, Button *button);
void handle_mouse_event(SDL_Event *e);
void predict_action();
void clear_action();
void exit_action();

int main(int argc, char *argv[]) {
    // Load the pre-trained model
    load_model("trained_model.bin");

    // Initialize SDL2
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        printf("SDL_Init Error: %s\n", SDL_GetError());
        return 1;
    }

    // Initialize SDL_ttf
    if (TTF_Init() != 0) {
        printf("TTF_Init Error: %s\n", TTF_GetError());
        SDL_Quit();
        return 1;
    }

    // Create window and renderer
    SDL_Window *win = SDL_CreateWindow("MNIST Digit Recognizer",
                                       SDL_WINDOWPOS_CENTERED,
                                       SDL_WINDOWPOS_CENTERED,
                                       WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_SHOWN);
    if (win == NULL) {
        printf("SDL_CreateWindow Error: %s\n", SDL_GetError());
        SDL_Quit();
        TTF_Quit();
        return 1;
    }

    SDL_Renderer *renderer = SDL_CreateRenderer(win, -1,
                                                SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (renderer == NULL) {
        SDL_DestroyWindow(win);
        printf("SDL_CreateRenderer Error: %s\n", SDL_GetError());
        SDL_Quit();
        TTF_Quit();
        return 1;
    }

    // Load font for rendering text
    // Update the font path if necessary
    TTF_Font *font = TTF_OpenFont("/usr/share/fonts/TTF/JetBrainsMonoNerdFont-Regular.ttf", 24);
    if (!font) {
        printf("TTF_OpenFont Error: %s\n", TTF_GetError());
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(win);
        SDL_Quit();
        TTF_Quit();
        return 1;
    }

    // Initialize grid and last_update arrays
    memset(grid, 0, sizeof(grid));
    memset(last_update, 0, sizeof(last_update));

    // Initialize buttons
    init_buttons();

    // Main loop
    SDL_Event e;
    Uint32 last_time = SDL_GetTicks();

    while (!quit) {
        // Event handling
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) {
                quit = 1;
            } else {
                handle_mouse_event(&e);
                // Handle key events
                if (e.type == SDL_KEYDOWN) {
                    if (e.key.keysym.sym == SDLK_e) {
                        current_tool = ERASE;
                    } else if (e.key.keysym.sym == SDLK_d) {
                        current_tool = DRAW;
                    }
                }
            }
        }

        // If drawing is active, update the grid gradually
        if (is_drawing) {
            int grid_x = current_mouse_x / CELL_SIZE;
            int grid_y = current_mouse_y / CELL_SIZE;

            if (grid_x >= 0 && grid_x < GRID_SIZE && grid_y >= 0 && grid_y < GRID_SIZE) {
                Uint32 current_ticks = SDL_GetTicks();
                Uint32 elapsed = current_ticks - last_update[grid_y][grid_x];

                if (elapsed >= UPDATE_INTERVAL) {
                    if (current_tool == DRAW) {
                        if (grid[grid_y][grid_x] <= 255 - STEP_VALUE) {
                            grid[grid_y][grid_x] += STEP_VALUE; // Increment pixel value by 5
                        } else {
                            grid[grid_y][grid_x] = 255; // Clamp to maximum
                        }
                    } else if (current_tool == ERASE) {
                        if (grid[grid_y][grid_x] >= STEP_VALUE) {
                            grid[grid_y][grid_x] -= STEP_VALUE; // Decrement pixel value by 5
                        } else {
                            grid[grid_y][grid_x] = 0; // Clamp to minimum
                        }
                    }
                    last_update[grid_y][grid_x] = current_ticks;
                }
            }
        }

        // Rendering
        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255); // White background
        SDL_RenderClear(renderer);

        // Render the grid
        render_grid(renderer, grid);

        // Render buttons
        render_button(renderer, font, &predict_button);
        render_button(renderer, font, &clear_button);
        render_button(renderer, font, &exit_button);

        // Render text (prediction)
        if (predicted_label != -1) {
            char text[100];
            sprintf(text, "Predicted Digit: %d (%.2f%% confidence)", predicted_label, confidence * 100);

            SDL_Color textColor = {0, 0, 0, 255}; // Black color
            SDL_Surface *textSurface = TTF_RenderText_Blended(font, text, textColor);
            if (!textSurface) {
                printf("TTF_RenderText_Blended Error: %s\n", TTF_GetError());
            } else {
                SDL_Texture *textTexture = SDL_CreateTextureFromSurface(renderer, textSurface);
                SDL_Rect textRect;
                textRect.x = 10;
                textRect.y = GRID_SIZE * CELL_SIZE + BUTTON_HEIGHT + 20;
                textRect.w = textSurface->w;
                textRect.h = textSurface->h;

                SDL_RenderCopy(renderer, textTexture, NULL, &textRect);

                SDL_FreeSurface(textSurface);
                SDL_DestroyTexture(textTexture);
            }
        }

        SDL_RenderPresent(renderer);

        // Frame rate control
        Uint32 current_frame_time = SDL_GetTicks();
        if (current_frame_time - last_time < 16) { // Approximately 60 FPS
            SDL_Delay(16 - (current_frame_time - last_time));
        }
        last_time = current_frame_time;
    }

    // Cleanup
    TTF_CloseFont(font);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(win);
    TTF_Quit();
    SDL_Quit();
    return 0;
}

// Function to load the trained model (weights and biases) from a file
void load_model(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Could not open file '%s' for loading weights!\n", filename);
        exit(1);
    }
    size_t read_count;

    read_count = fread(weights_input_hidden, sizeof(double), INPUT_SIZE * HIDDEN_SIZE, file);
    if (read_count != INPUT_SIZE * HIDDEN_SIZE) {
        printf("Error: Failed to read weights_input_hidden from '%s'\n", filename);
        fclose(file);
        exit(1);
    }

    read_count = fread(weights_hidden_output, sizeof(double), HIDDEN_SIZE * OUTPUT_SIZE, file);
    if (read_count != HIDDEN_SIZE * OUTPUT_SIZE) {
        printf("Error: Failed to read weights_hidden_output from '%s'\n", filename);
        fclose(file);
        exit(1);
    }

    read_count = fread(bias_hidden, sizeof(double), HIDDEN_SIZE, file);
    if (read_count != HIDDEN_SIZE) {
        printf("Error: Failed to read bias_hidden from '%s'\n", filename);
        fclose(file);
        exit(1);
    }

    read_count = fread(bias_output, sizeof(double), OUTPUT_SIZE, file);
    if (read_count != OUTPUT_SIZE) {
        printf("Error: Failed to read bias_output from '%s'\n", filename);
        fclose(file);
        exit(1);
    }

    fclose(file);
    printf("Model loaded from '%s'\n", filename);
}

// ReLU activation function
double relu(double x) {
    return x > 0 ? x : 0;
}

// Softmax function for output layer probabilities
void softmax(double *output, int size) {
    double sum = 0.0;
    double max_logit = output[0];

    // Find the max logit to prevent overflow
    for (int i = 1; i < size; i++) {
        if (output[i] > max_logit) {
            max_logit = output[i];
        }
    }

    // Compute exponentials and sum
    for (int i = 0; i < size; i++) {
        output[i] = exp(output[i] - max_logit); // Subtract max_logit for numerical stability
        sum += output[i];
    }

    // Normalize
    for (int i = 0; i < size; i++) {
        output[i] /= sum;
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
int predict(double *input, double *confidence_out) {
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
    *confidence_out = max_prob;
    return predicted_label;
}

// Function to predict based on the grid
int predict_grid(Uint8 grid[GRID_SIZE][GRID_SIZE], double *confidence_out) {
    double input[INPUT_SIZE];
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            input[i * GRID_SIZE + j] = grid[i][j] / 255.0; // Normalize pixel values
        }
    }
    return predict(input, confidence_out);
}

// Initialize buttons
void init_buttons() {
    // Predict Button
    predict_button.rect.x = 10;
    predict_button.rect.y = GRID_SIZE * CELL_SIZE + 10;
    predict_button.rect.w = BUTTON_WIDTH;
    predict_button.rect.h = BUTTON_HEIGHT;
    predict_button.color = (SDL_Color){0, 122, 204, 255}; // Blue color
    strcpy(predict_button.label, "Predict");
    predict_button.action = predict_action;

    // Clear Button
    clear_button.rect.x = 120;
    clear_button.rect.y = GRID_SIZE * CELL_SIZE + 10;
    clear_button.rect.w = BUTTON_WIDTH;
    clear_button.rect.h = BUTTON_HEIGHT;
    clear_button.color = (SDL_Color){204, 204, 0, 255}; // Yellow color
    strcpy(clear_button.label, "Clear");
    clear_button.action = clear_action;

    // Exit Button
    exit_button.rect.x = 230;
    exit_button.rect.y = GRID_SIZE * CELL_SIZE + 10;
    exit_button.rect.w = BUTTON_WIDTH;
    exit_button.rect.h = BUTTON_HEIGHT;
    exit_button.color = (SDL_Color){204, 0, 0, 255}; // Red color
    strcpy(exit_button.label, "Exit");
    exit_button.action = exit_action;
}

// Render a button
void render_button(SDL_Renderer *renderer, TTF_Font *font, Button *button) {
    // Draw button rectangle
    SDL_SetRenderDrawColor(renderer, button->color.r, button->color.g, button->color.b, button->color.a);
    SDL_RenderFillRect(renderer, &button->rect);

    // Draw button label
    SDL_Color textColor = {255, 255, 255, 255}; // White color
    SDL_Surface *textSurface = TTF_RenderText_Blended(font, button->label, textColor);
    if (textSurface) {
        SDL_Texture *textTexture = SDL_CreateTextureFromSurface(renderer, textSurface);

        // Center the text in the button
        SDL_Rect textRect;
        textRect.w = textSurface->w;
        textRect.h = textSurface->h;
        textRect.x = button->rect.x + (button->rect.w - textRect.w) / 2;
        textRect.y = button->rect.y + (button->rect.h - textRect.h) / 2;

        SDL_RenderCopy(renderer, textTexture, NULL, &textRect);

        SDL_FreeSurface(textSurface);
        SDL_DestroyTexture(textTexture);
    }
}

// Handle mouse events with drawing state tracking
void handle_mouse_event(SDL_Event *e) {
    int x, y;

    if (e->type == SDL_MOUSEBUTTONDOWN) {
        x = e->button.x;
        y = e->button.y;

        // Check if the Predict button is clicked
        if (SDL_PointInRect(&(SDL_Point){x, y}, &predict_button.rect)) {
            predict_button.action();
            return;
        }
        // Check if the Clear button is clicked
        else if (SDL_PointInRect(&(SDL_Point){x, y}, &clear_button.rect)) {
            clear_button.action();
            return;
        }
        // Check if the Exit button is clicked
        else if (SDL_PointInRect(&(SDL_Point){x, y}, &exit_button.rect)) {
            exit_button.action();
            return;
        }
        // Handle drawing and erasing
        else if (e->button.button == SDL_BUTTON_LEFT) {
            is_drawing = 1; // Start drawing
            current_mouse_x = x;
            current_mouse_y = y;
        }
    } 
    else if (e->type == SDL_MOUSEBUTTONUP) {
        if (e->button.button == SDL_BUTTON_LEFT) {
            is_drawing = 0; // Stop drawing
        }
    } 
    else if (e->type == SDL_MOUSEMOTION) {
        if (e->motion.state & SDL_BUTTON_LMASK) {
            current_mouse_x = e->motion.x;
            current_mouse_y = e->motion.y;
        }
    }
}

// Button actions
void predict_action() {
    predicted_label = predict_grid(grid, &confidence);
}

void clear_action() {
    memset(grid, 0, sizeof(grid)); // Clear grid
    memset(last_update, 0, sizeof(last_update)); // Reset last_update
    predicted_label = -1; // Reset prediction
}

void exit_action() {
    quit = 1;
}

// Render the grid with updated pixel values
void render_grid(SDL_Renderer *renderer, Uint8 grid[GRID_SIZE][GRID_SIZE]) {
    SDL_Rect cell;
    cell.w = CELL_SIZE;
    cell.h = CELL_SIZE;

    for (int y = 0; y < GRID_SIZE; y++) {
        for (int x = 0; x < GRID_SIZE; x++) {
            Uint8 intensity = grid[y][x];
            SDL_SetRenderDrawColor(renderer, intensity, intensity, intensity, 255);
            cell.x = x * CELL_SIZE;
            cell.y = y * CELL_SIZE;
            SDL_RenderFillRect(renderer, &cell);
        }
    }

    // Optionally, draw grid lines for better visibility
    SDL_SetRenderDrawColor(renderer, 200, 200, 200, 255); // Light gray color
    for (int i = 0; i <= GRID_SIZE; i++) {
        // Vertical lines
        SDL_RenderDrawLine(renderer, i * CELL_SIZE, 0, i * CELL_SIZE, GRID_SIZE * CELL_SIZE);
        // Horizontal lines
        SDL_RenderDrawLine(renderer, 0, i * CELL_SIZE, GRID_SIZE * CELL_SIZE, i * CELL_SIZE);
    }
}

