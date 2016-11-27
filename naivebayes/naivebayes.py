import sys
import os.path
import numpy as np
from collections import defaultdict

import util

USAGE = "%s <test data folder> <spam folder> <ham folder>"

def get_counts(file_list):
    """
    Computes counts for each word that occurs in the files in file_list.

    Inputs
    ------
    file_list : a list of filenames, suitable for use with open() or 
                util.get_words_in_file()

    Output
    ------
    A dict whose keys are words, and whose values are the number of files the
    key occurred in.
    """
    cnt = defaultdict(lambda: 0)
    for filename in file_list:
        words = set(util.get_words_in_file(filename))
        for word in words:
            cnt[word] += 1
    return cnt

def get_log_probabilities(file_list):
    """
    Computes log-frequencies for each word that occurs in the files in 
    file_list.

    Input
    -----
    file_list : a list of filenames, suitable for use with open() or 
                util.get_words_in_file()

    Output
    ------
    A dict whose keys are words, and whose values are the log of the smoothed
    estimate of the fraction of files the key occurred in.

    Hint
    ----
    The data structure util.DefaultDict will be useful to you here, as will the
    get_counts() helper above.
    """
    num_files = len(file_list)
    cnt = get_counts(file_list)
    log_probs = {}
    for k, v in cnt.items():
        log_probs[k] = np.log(v + 1) -np.log(num_files + 2)

    return log_probs
    
def learn_distributions(file_lists_by_category, n_spams, n_hams):
    """
    Input
    -----
    A two-element list. The first element is a list of spam files, 
    and the second element is a list of ham (non-spam) files.

    Output
    ------
    (log_probabilities_by_category, log_prior)

    log_probabilities_by_category : A list whose first element is a smoothed
                                    estimate for log P(y=w_j|c=spam) (as a dict,
                                    just as in get_log_probabilities above), and
                                    whose second element is the same for c=ham.

    log_prior_by_category : A list of estimates for the log-probabilities for
                            each class:
                            [est. for log P(c=spam), est. for log P(c=ham)]
    """
    log_probabilities_by_category = []

    for category_list in file_lists_by_category:
        log_probabilities_by_category.append(get_log_probabilities(category_list))
    
    log_prior_by_category = [np.log(n_spams) - np.log(n_spams+n_hams), np.log(n_hams) - np.log(n_spams+n_hams)]
    return log_probabilities_by_category, log_prior_by_category

def classify_email(email_filename,
                   log_probabilities_by_category,
                   log_prior_by_category,
                   n_spams, n_hams):
    """
    Uses Naive Bayes classification to classify the email in the given file.

    Inputs
    ------
    email_filename : name of the file containing the email to be classified

    log_probabilities_by_category : See output of learn_distributions

    log_prior_by_category : See output of learn_distributions

    Output
    ------
    One of the labels in names.
    """
    words = util.get_words_in_file(email_filename)
    log_prob_spam = log_prior_by_category[0]
    log_prob_ham = log_prior_by_category[1]
    for word in words:
        if word in log_probabilities_by_category[0]:
            log_prob_spam += log_probabilities_by_category[0][word]
        else:
            log_prob_spam += np.log(1/(n_spams+2))

        if word in log_probabilities_by_category[1]:    
            log_prob_ham += log_probabilities_by_category[1][word]
        else:
            log_prob_ham += np.log(1/(n_hams+2))

    if log_prob_spam > log_prob_ham:
        return "spam"
    else:
        return "ham"


def classify_emails(spam_files, ham_files, test_files):
    # DO NOT MODIFY -- used by the autograder
    log_probabilities_by_category, log_prior = \
        learn_distributions([spam_files, ham_files])
    estimated_labels = []
    for test_file in test_files:
        estimated_label = \
            classify_email(test_file, log_probabilities_by_category, log_prior)
        estimated_labels.append(estimated_label)
    return estimated_labels

def main():
    ### Read arguments
    if len(sys.argv) != 4:
        print(USAGE % sys.argv[0])
    testing_folder = sys.argv[1]
    (spam_folder, ham_folder) = sys.argv[2:4]

    ### Learn the distributions
    file_lists = []
    for folder in (spam_folder, ham_folder):
        file_lists.append(util.get_files_in_folder(folder))
    n_spams = len(file_lists[0])
    n_hams = len(file_lists[1])
    (log_probabilities_by_category, log_priors_by_category) = \
            learn_distributions(file_lists, n_spams, n_hams)


    # Here, columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    performance_measures = np.zeros([2,2])

    ### Classify and measure performance
    for filename in (util.get_files_in_folder(testing_folder)):
        ## Classify
        label = classify_email(filename,
                               log_probabilities_by_category,
                               log_priors_by_category,
                               n_spams, n_hams)
        ## Measure performance
        # Use the filename to determine the true label
        base = os.path.basename(filename)
        true_index = ('ham' in base)
        guessed_index = (label == 'ham')
        performance_measures[true_index, guessed_index] += 1


        # Uncomment this line to see which files your classifier
        # gets right/wrong:
        #print("%s : %s" %(label, filename))

    template="You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
    # Correct counts are on the diagonal
    correct = np.diag(performance_measures)
    # totals are obtained by summing across guessed labels
    totals = np.sum(performance_measures, 1)
    print(template % (correct[0],
                      totals[0],
                      correct[1],
                      totals[1]))

if __name__ == '__main__':
    main()
