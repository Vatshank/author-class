# author-class
Classification of authors

A simple classifier for predicting the author of the book given the text as input with a goal of using simple features which are less expensive to compute. 

I avoided using n-grams as features as they take time to compute for longer texts. Instead simple features which are characteristic of the vocabulary of the author are used (like the number of unique words, the number of words used exactly once, distribution of the length of the sentences, the distribution of the length of the words, the distribution of number of pronouns/conjunctions per sentence). This gives a total of 83 numeric features (each of the distribution contributes 20 features). A Random forest classifier is then used for the classification task.
