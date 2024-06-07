#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <argp.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>

#define CHUNK_SIZE 512
#define DEFAULT_VOCABULARY_SIZE 50257

typedef struct
{
    int *tokens;
    size_t size;
    size_t capacity;
} Vocabulary;

typedef struct
{
    Vocabulary *vocabulary;
    int *token_frequencies;
} BPETokenizer;

typedef struct
{
    char *action;
    char *training_dataset;
    char *training_output;
    int vocabulary_size;
    char *tokenizer_data;
    char *run_data;
    int in_place;
} Arguments;

static int arg_parse(int key, char *arg, struct argp_state *state);
static void dict_to_defaultdict(int *dict, size_t dict_size, int *default_dict, size_t default_dict_size);
static void duplicate_file(const char *source_path, const char *destination_path, size_t chunk_size);
static ssize_t read_utf_8_chunk(int fd, char *buffer, size_t chunk_size);
static ssize_t read_binary_chunk(int fd, uint32_t *buffer, size_t chunk_size);
static void bpe_tokenizer_train(BPETokenizer *tokenizer, const char *dataset_path, size_t vocabulary_size, int in_place);
static int all_pairs_are_unique(int *token_frequencies, size_t size);
static void count_token_frequencies(BPETokenizer *tokenizer, uint32_t *tokens, size_t size);
static void sort_by_token_frequency(int *token_frequencies, size_t size, int *sorted_pairs);
static void bpe_tokenizer_tokenize(BPETokenizer *tokenizer, const char *data, int *tokens);
static void bpe_tokenizer_detokenize(BPETokenizer *tokenizer, int *tokens, size_t size, char *output);

const char *argp_program_version = "bpe_tokenizer 1.0";
const char *argp_program_bug_address = "<bug@example.com>";
static char doc[] = "Byte-Pair Encoding Tokenizer";
static char args_doc[] = "";
static struct argp_option options[] = {
    {"action", 'a', "ACTION", 0, "Action to perform (train, tokenize, detokenize)"},
    {"training_dataset", 'd', "DATASET", 0, "Training dataset file path"},
    {"training_output", 'o', "OUTPUT", 0, "Training output file path"},
    {"vocabulary_size", 'v', "SIZE", 0, "Vocabulary size"},
    {"tokenizer_data", 't', "TOKENIZER", 0, "Tokenizer data file path"},
    {"run_data", 'r', "RUNDATA", 0, "Run data file path"},
    {"in_place", 'i', 0, 0, "In-place modification"},
    {0}};

static struct argp argp = {options, arg_parse, args_doc, doc};

static int arg_parse(int key, char *arg, struct argp_state *state)
{
    Arguments *arguments = state->input;

    switch (key)
    {
    case 'a':
        arguments->action = arg;
        break;
    case 'd':
        arguments->training_dataset = arg;
        break;
    case 'o':
        arguments->training_output = arg;
        break;
    case 'v':
        arguments->vocabulary_size = atoi(arg);
        break;
    case 't':
        arguments->tokenizer_data = arg;
        break;
    case 'r':
        arguments->run_data = arg;
        break;
    case 'i':
        arguments->in_place = 1;
        break;
    case ARGP_KEY_ARG:
        if (state->arg_num >= 1)
            argp_usage(state);
        break;
    case ARGP_KEY_END:
        if (state->arg_num < 1)
            argp_usage(state);
        break;
    default:
        return ARGP_ERR_UNKNOWN;
    }
    return 0;
}

static void dict_to_defaultdict(int *dict, size_t dict_size, int *default_dict, size_t default_dict_size)
{
    memset(default_dict, 0, sizeof(int) * default_dict_size);
    for (size_t i = 0; i < dict_size; ++i)
    {
        default_dict[i] = dict[i];
    }
}

static void duplicate_file(const char *source_path, const char *destination_path, size_t chunk_size)
{
    int source_fd = open(source_path, O_RDONLY);
    int dest_fd = open(destination_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);

    if (source_fd == -1 || dest_fd == -1)
    {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    char *buffer = malloc(chunk_size);
    if (!buffer)
    {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }

    ssize_t bytes_read;
    while ((bytes_read = read(source_fd, buffer, chunk_size)) > 0)
    {
        if (write(dest_fd, buffer, bytes_read) != bytes_read)
        {
            perror("Error writing to file");
            exit(EXIT_FAILURE);
        }
    }

    free(buffer);
    close(source_fd);
    close(dest_fd);
}

static ssize_t read_utf_8_chunk(int fd, char *buffer, size_t chunk_size)
{
    return read(fd, buffer, chunk_size);
}

static ssize_t read_binary_chunk(int fd, uint32_t *buffer, size_t chunk_size)
{
    return read(fd, buffer, chunk_size * sizeof(uint32_t)) / sizeof(uint32_t);
}

static void bpe_tokenizer_train(BPETokenizer *tokenizer, const char *dataset_path, size_t vocabulary_size, int in_place)
{
    if (!dataset_path)
    {
        fprintf(stderr, "Please specify a path to a local file containing dataset\n");
        exit(EXIT_FAILURE);
    }

    char working_copy_path[256];
    if (in_place)
    {
        strcpy(working_copy_path, dataset_path);
    }
    else
    {
        strcpy(working_copy_path, "working_copy.txt");
        duplicate_file(dataset_path, working_copy_path, CHUNK_SIZE);
    }

    int working_copy_fd = open(working_copy_path, O_RDWR);
    if (working_copy_fd == -1)
    {
        perror("Error opening working copy file");
        exit(EXIT_FAILURE);
    }

    char buffer[CHUNK_SIZE];
    ssize_t bytes_read;

    while ((bytes_read = read_utf_8_chunk(working_copy_fd, buffer, CHUNK_SIZE)) > 0)
    {
        for (ssize_t i = 0; i < bytes_read; ++i)
        {
            // Process each character
        }
    }

    // Implement the training loop

    close(working_copy_fd);
}

static int all_pairs_are_unique(int *token_frequencies, size_t size)
{
    for (size_t i = 0; i < size; ++i)
    {
        if (token_frequencies[i] > 1)
            return 0;
    }
    return 1;
}

static void count_token_frequencies(BPETokenizer *tokenizer, uint32_t *tokens, size_t size)
{
    for (size_t i = 0; i < size - 1; i += 2)
    {
        int pair_index = tokens[i] * 256 + tokens[i + 1];
        tokenizer->token_frequencies[pair_index]++;
    }
}

static void sort_by_token_frequency(int *token_frequencies, size_t size, int *sorted_pairs)
{
    // Implement a sorting function
}

void bpe_tokenizer_tokenize(BPETokenizer *tokenizer, const char *data, int *tokens)
{
    // Implement tokenization
    size_t data_length = strlen(data);
    tokens = malloc(data_length * sizeof(int));

    for (size_t i = 0; i < data_length; ++i)
    {
        tokens[i] = (int)data[i];
    }

    int added_keys[256]; // Adjust size according to your needs
    size_t added_keys_count = 0;

    for (size_t i = 256; i < tokenizer->vocabulary->size; ++i)
    {
        added_keys[added_keys_count++] = i;
    }

    for (size_t i = 0; i < added_keys_count; ++i)
    {
        int vocabulary_item = added_keys[i];
        int *new_tokens = malloc(data_length * sizeof(int));
        size_t new_tokens_count = 0;
        size_t index = 0;

        while (index < data_length)
        {
            if (index + 1 < data_length &&
                tokenizer->vocabulary->tokens[vocabulary_item * 2] == tokens[index] &&
                tokenizer->vocabulary->tokens[vocabulary_item * 2 + 1] == tokens[index + 1])
            {
                new_tokens[new_tokens_count++] = vocabulary_item;
                index += 2;
            }
            else
            {
                new_tokens[new_tokens_count++] = tokens[index++];
            }
        }

        free(tokens);
        tokens = new_tokens;
        data_length = new_tokens_count;
    }

    for (size_t i = 0; i < data_length; ++i)
    {
        printf("%d ", tokens[i]);
    }
    printf("\n");

    free(tokens);
}

void bpe_tokenizer_detokenize(BPETokenizer *tokenizer, int *tokens, size_t size, char *output)
{
    // Implement detokenization
    size_t output_length = 0;
    for (size_t i = 0; i < size; ++i)
    {
        if (tokens[i] < 256)
        {
            output[output_length++] = (char)tokens[i];
        }
        else
        {
            output[output_length++] = (char)tokenizer->vocabulary->tokens[tokens[i] * 2];
            output[output_length++] = (char)tokenizer->vocabulary->tokens[tokens[i] * 2 + 1];
        }
    }
    output[output_length] = '\0';
}

int main(int argc, char **argv)
{
    Arguments arguments = {NULL, NULL, NULL, DEFAULT_VOCABULARY_SIZE, NULL, NULL, 0};
    argp_parse(&argp, argc, argv, 0, 0, &arguments);

    BPETokenizer tokenizer;
    int vocabulary[DEFAULT_VOCABULARY_SIZE];
    int token_frequencies[DEFAULT_VOCABULARY_SIZE * DEFAULT_VOCABULARY_SIZE] = {0};

    tokenizer.vocabulary = (Vocabulary *)malloc(sizeof(Vocabulary));
    tokenizer.vocabulary->tokens = vocabulary;
    tokenizer.vocabulary->size = 0;
    tokenizer.vocabulary->capacity = DEFAULT_VOCABULARY_SIZE;
    tokenizer.token_frequencies = token_frequencies;

    if (strcmp(arguments.action, "train") == 0)
    {
        bpe_tokenizer_train(&tokenizer, arguments.training_dataset, arguments.vocabulary_size, arguments.in_place);
    }
    else if (strcmp(arguments.action, "tokenize") == 0)
    {
        int *tokens = NULL;
        bpe_tokenizer_tokenize(&tokenizer, arguments.run_data, tokens);
    }
    else if (strcmp(arguments.action, "detokenize") == 0)
    {
        int tokens[CHUNK_SIZE]; // adjust this
        size_t token_count = 0;
        // Read tokens from file or another source and set token_count
        char output[CHUNK_SIZE * 2]; // Adjust based on maximum possible output size
        bpe_tokenizer_detokenize(&tokenizer, tokens, token_count, output);
        printf("%s\n", output);
    }

    free(tokenizer.vocabulary);
    return 0;
}
