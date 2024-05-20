'''
Main script for testing the assignment.
Runs the tests on the results json file.
'''

import argparse
import json

def get_args():
    parser = argparse.ArgumentParser(description='Language Modeling')
    parser.add_argument('test', type=str, help='The test to perform.')
    return parser.parse_args()

def test_read_data_data_types(results):
    is_data_list = results["is_data_list"]
    is_data_first_element_list = results["is_data_first_element_list"]
    is_data_first_element_first_item_tuple = results["is_data_first_element_first_item_tuple"]

    if not is_data_list:
        return f"Data is not a list"
    if not is_data_first_element_list:
        return f"First element of data is not a list"
    if not is_data_first_element_first_item_tuple:
        return f"First element of first element of data is not a tuple"
    return 1

def test_read_data_len_train_data(results):
    if results["train_data_length"] != 9521:
        return f"Vocab length is {results['vocab_length']}, expected 9521"
    return 1

def test_generate_vocabs(results):
    if results["vocab_size"] not in [19873, 19874]:
        return f"Vocab length is {results['vocab_size']}, expected 19873/19874"
    if results["num_tags"] != 18:
        return f"Number of tags is {results['num_tags']}, expected 18"
    return 1
    
def test_hmm(results):
    precision = results["precision"]
    recall = results["recall"]
    f1 = results["f1"]

    # Min values for precision, recall and f1 to pass
    if precision < 0.7:
        return f"Precision is {precision}, expected at least 0.7"
    if recall < 0.62:
        return f"Recall is {recall}, expected at least 0.62"
    if f1 < 0.65:
        return f"F1 is {f1}, expected at least 0.65"
    
    # Values to partially pass
    if precision < 0.76:
        return 2
    if recall < 0.68:
        return 2
    if f1 < 0.7:
        return 2
    # Pass with full marks
    return 1

def test_embeddings_model(results):
    precision = results["precision"]
    recall = results["recall"]
    f1 = results["f1"]

    # Min values for precision, recall and f1 to pass
    if precision < 0.72:
        return f"Precision is {precision}, expected at least 0.7"
    if recall < 0.70:
        return f"Recall is {recall}, expected at least 0.62"
    if f1 < 0.70:
        return f"F1 is {f1}, expected at least 0.65"
    
    # Values to partially pass
    if precision < 0.77:
        return 2
    if recall < 0.76:
        return 2
    if f1 < 0.76:
        return 2
    # Pass with full marks
    return 1

def test_nltk_tagger(results):
    accuracy = results["accuracy"]

    # Min values for accuracy and f_measure to pass
    if accuracy < 0.80:
        return f"Accuracy is {accuracy}, expected at least 0.83"

    # Values to partially pass
    if accuracy < 0.83:
        return 2

    # Pass with full marks
    return 1


def main():
    # Get command line arguments
    args = get_args()

    # Read results.json
    with open('results.json', 'r') as f:
        results = json.load(f)

    # Initialize the result variable
    result = None

    # Switch between the tests
    match args.test:
        case 'test_read_data_data_types':
            result = test_read_data_data_types(results["test_read_data_data_types"])
        case 'test_read_data_len_train_data':
            result = test_read_data_len_train_data(results["test_read_data_len_train_data"])
        case 'test_generate_vocabs':
            result = test_generate_vocabs(results["test_generate_vocabs"])
        case 'test_hmm':
            result = test_hmm(results["test_hmm"])
        case 'test_embeddings_model':
            result = test_embeddings_model(results["test_embeddings_model"])
        case 'test_nltk_tagger':
            result = test_nltk_tagger(results["test_nltk_tagger"])
        case _:
            print('Invalid test.')

    # Print the result for the autograder to capture
    if result is not None:
        print(result)

if __name__ == '__main__':
    main()
