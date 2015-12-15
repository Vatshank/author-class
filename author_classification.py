import glob
from collections import Counter
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk import tag
from sklearn.ensemble import RandomForestClassifier


def get_features(in_path):
    print in_path
    features = []
    with open(in_path) as f_in:
        text = f_in.read().decode("utf-8")

    # Sentence length distribution
    sentences = sent_tokenize(text)
    len_sentences = [len(sent) for sent in sentences]
    counter_len_sentences = Counter(len_sentences)
    distribution_len_sentences = [counter_len_sentences[key] for key in range(1, 20)]
    distribution_len_sentences.append(sum([counter_len_sentences[key] for key in counter_len_sentences.keys() if key>=20]))
    features.extend(distribution_len_sentences)

    # Word distribution
    words = [word for sentence in sentences for word in word_tokenize(sentence.strip())]
    counter_words = Counter(words)
    num_unique_words = len(set(words))
    num_freq_1 = sum(1 for val in counter_words.values() if val==1)
    num_freq_2 = sum(1 for val in counter_words.values() if val==2)
    features.extend([num_unique_words, num_freq_1, num_freq_2])

    # Word length distribution
    len_words = [len(word) for word in words]
    counter_len_words = Counter(len_words)
    distribution_len_words = [counter_len_words[key] for key in range(1, 20)]
    distribution_len_words.append(sum([counter_len_words[key] for key in counter_len_words.keys() if key>=20]))
    features.extend(distribution_len_words)

    # Pronouns and conjunctions distribution
    tagger = tag.PerceptronTagger()
    tagset = None
    pos_tags = [tag._pos_tag(word_tokenize(sentence), tagset, tagger) for sentence in sentences]
    pronoun_set = {"PRP", "PRP$", "WP", "WP$"}
    conjunction_set = {"CC", "IN"}
    counter_pronouns = Counter([sum(1 for tag in tags if tag[1] in pronoun_set) for tags in pos_tags])
    counter_conjunctions = Counter([sum(1 for tag in tags if tag[1] in conjunction_set) for tags in pos_tags])
    distribution_pronouns = [counter_pronouns[key] for key in range(1, 20)]
    distribution_pronouns.append(sum([counter_pronouns[key] for key in counter_pronouns.keys() if key>=20]))
    distribution_conjunctions = [counter_conjunctions[key] for key in range(1, 20)]
    distribution_conjunctions.append(sum([counter_conjunctions[key] for key in counter_conjunctions.keys() if key>=20]))
    features.extend(distribution_pronouns)
    features.extend(distribution_conjunctions)

    return features

if __name__ == '__main__':
    books_train = glob.glob("/users/vatshank/Downloads/books/training/*.txt")
    books_test = glob.glob("/users/vatshank/Downloads/books/testing/*.txt")

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    # Training set
    for path in books_train:
        y_train.append(path.strip().split("/")[-1].split("_")[0])
        x_train.append(get_features(path))

    # Testing set
    for path in books_test:
        y_test.append(path.strip().split("/")[-1].split("_")[0])
        x_test.append(get_features(path))

    # Classification
    clf = RandomForestClassifier(n_estimators=50, class_weight="balanced")
    clf.fit(x_train, y_train)
    print sum(y_test==clf.predict(x_test))
