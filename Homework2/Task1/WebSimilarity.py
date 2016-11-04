# Post-process the crawled music context data, extract term weights, and compute cosine similarities.
__author__ = 'mms'

import os
import numpy as np
import scipy.spatial.distance as scidist      # import distance computation module from scipy package
import urllib
import Wikipedia_Fetcher
import helper # helper.py
import Musixmatch_fetcher as mf
import json
import re
from langdetect import detect
from nltk.stem import PorterStemmer


# Parameters
#WIKIPEDIA_TFIDFS = "./tfidfs_100u.txt"            # file to store term weights
#WIKIPEDIA_TERMS = "./terms_100u.txt"             # file to store list of terms (for easy interpretation of term weights)
#WIKIPEDIA_AAM = "./AAM_100u.txt"               # file to store similarities between items
OUTPUT_DIR = './output/'
WIKIPEDIA_OUTPUT = OUTPUT_DIR + 'wikipedia/'
WIKIPEDIA_TFIDFS = WIKIPEDIA_OUTPUT + "tfidfs.txt"            # file to store term weights
WIKIPEDIA_TERMS = WIKIPEDIA_OUTPUT + "terms.txt"             # file to store list of terms (for easy interpretation of term weights)
WIKIPEDIA_AAM = WIKIPEDIA_OUTPUT + "AAM.txt"               # file to store similarities between items

MUSIXMATCH_OUTPUT = OUTPUT_DIR + 'musixmatch/'
MUSIXMATCH_TFIDFS = MUSIXMATCH_OUTPUT + "tfidfs.txt"            # file to store term weights
MUSIXMATCH_TERMS = MUSIXMATCH_OUTPUT + "terms.txt"             # file to store list of terms (for easy interpretation of term weights)
MUSIXMATCH_AAM = MUSIXMATCH_OUTPUT + "AAM.txt"               # file to store similarities between items
MUSIXMATCH_AAM_ARTIST_ID = "artist_id_aam.txt"
MUSIXMATCH_ARTISTS_ID = MUSIXMATCH_OUTPUT + 'artist_ids.txt'

# Stop words used by Google
STOP_WORDS = ["a", "able", "about", "above", "abroad", "according", "accordingly", "across", "actually", "adj", "after", "afterwards", "again", "against", "ago", "ahead", "ain't", "all",
    "allow", "allows", "almost", "alone", "along", "alongside", "already", "also", "although", "always", "am", "amid", "amidst", "among", "amongst", "an", "and", "another", "any", "anybody",
    "anyhow", "anyone", "anything", "anyway", "anyways", "anywhere", "apart", "appear", "appreciate", "appropriate", "are", "aren't", "around", "as", "a's", "aside", "ask", "asking", "associated", "at", "available", "away",
    "awfully", "b", "back", "backward", "backwards", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "begin", "behind", "being", "believe", "below", "beside", "besides", "best", "better",
    "between", "beyond", "both", "brief", "but", "by", "c", "came", "can", "cannot", "cant", "can't", "caption", "cause", "causes", "certain", "certainly", "changes", "clearly", "c'mon", "co", "co.", "com", "come",
    "comes", "concerning", "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding", "could", "couldn't", "course", "c's", "currently", "d", "dare", "daren't",
    "definitely", "described", "despite", "did", "didn't", "different", "directly", "do", "does", "doesn't", "doing", "done", "don't", "down", "downwards", "during", "e",
    "each", "edu", "eg", "eight", "eighty", "either", "else", "elsewhere", "end", "ending", "enough", "entirely", "especially", "et", "etc", "even", "ever", "evermore",
    "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "f", "fairly", "far", "farther", "few", "fewer", "fifth", "first", "five", "followed", "following", "follows", "for", "forever", "former",
    "formerly", "forth", "forward", "found", "four", "from", "further", "furthermore", "g", "get", "gets", "getting", "given", "gives", "go", "goes", "going", "gone", "got", "gotten", "greetings", "h", "had", "hadn't", "half", "happens",
    "hardly", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "hello", "help", "hence", "her", "here", "hereafter", "hereby", "herein", "here's", "hereupon", "hers", "herself", "he's", "hi", "him", "himself", "his",
    "hither", "hopefully", "how", "howbeit", "however", "hundred", "i", "i'd", "ie", "if", "ignored", "i'll", "i'm", "immediate", "in", "inasmuch", "inc", "inc.", "indeed", "indicate", "indicated", "indicates", "inner", "inside",
    "insofar", "instead", "into", "inward", "is", "isn't", "it", "it'd", "it'll", "its", "it's", "itself", "i've", "j", "just", "k", "keep", "keeps", "kept", "know", "known", "knows", "l", "last", "lately", "later", "latter",
    "latterly", "least", "less", "lest", "let", "let's", "like", "liked", "likely", "likewise", "little", "look", "looking", "looks", "low", "lower", "ltd", "m", "made", "mainly", "make", "makes", "many", "may", "maybe", "mayn't",
    "me", "mean", "meantime", "meanwhile", "merely", "might", "mightn't", "mine", "minus", "miss", "more", "moreover", "most", "mostly", "mr", "mrs", "much", "must", "mustn't", "my", "myself", "n", "name", "namely", "nd", "near",
    "nearly", "necessary", "need", "needn't", "needs", "neither", "never", "neverf", "neverless", "nevertheless", "new", "next", "nine", "ninety", "no", "nobody", "non", "none", "nonetheless", "noone", "no-one", "nor",
    "normally", "not", "nothing", "notwithstanding", "novel", "now", "nowhere", "o", "obviously", "of", "off", "often", "oh", "ok", "okay", "old", "on", "once", "one", "ones", "one's", "only", "onto", "opposite", "or",
    "other", "others", "otherwise", "ought", "oughtn't", "our", "ours", "ourselves", "out", "outside", "over", "overall", "own", "p", "particular", "particularly", "past", "per", "perhaps", "placed", "please", "plus",
    "possible", "presumably", "probably", "provided", "provides", "q", "que", "quite", "qv", "r", "rather", "rd", "re", "really", "reasonably", "recent", "recently", "regarding", "regardless", "regards", "relatively",
    "respectively", "right", "round", "s", "said", "same", "saw", "say", "saying", "says", "second", "secondly", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent",
    "serious", "seriously", "seven", "several", "shall", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "since", "six", "so", "some", "somebody", "someday", "somehow", "someone", "something",
    "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "specified", "specify", "specifying", "still", "sub", "such", "sup", "sure", "t", "take", "taken", "taking", "tell", "tends", "th", "than",
    "thank", "thanks", "thanx", "that", "that'll", "thats", "that's", "that've", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "there'd", "therefore",
    "therein", "there'll", "there're", "theres", "there's", "thereupon", "there've", "these", "they", "they'd", "they'll", "they're", "they've", "thing", "things", "think", "third", "thirty", "this",
    "thorough", "thoroughly", "those", "though", "three", "through", "throughout", "thru", "thus", "till", "to", "together", "too", "took", "toward", "towards", "tried", "tries", "truly", "try",
    "trying", "t's", "twice", "two", "u", "un", "under", "underneath", "undoing", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "up", "upon", "upwards", "us", "use", "used",
    "useful", "uses", "using", "usually", "v", "value", "various", "versus", "very", "via", "viz", "vs", "w", "want", "wants", "was", "wasn't", "way", "we", "we'd", "welcome", "well", "we'll",
    "went", "were", "we're", "weren't", "we've", "what", "whatever", "what'll", "what's", "what've", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein",
    "where's", "whereupon", "wherever", "whether", "which", "whichever", "while", "whilst", "whither", "who", "who'd", "whoever", "whole", "who'll", "whom", "whomever", "who's", "whose",
    "why", "will", "willing", "wish", "with", "within", "without", "wonder", "won't", "would", "wouldn't", "x", "y", "yes", "yet", "you", "you'd", "you'll", "your", "you're", "yours",
    "yourself", "yourselves", "you've", "z", "zero"
]

# A simple function to remove HTML tags from a string.
# You can of course also use some fancy library. In particular, lxml (http://lxml.de/) seems a simple and good solution; also for getting rid of javascript.
def remove_html_markup(s):
    tag = False
    quote = False
    out = ""
    # for all characters in string s
    for c in s:
        if c == '<' and not quote:
            tag = True
        elif c == '>' and not quote:
            tag = False
        elif (c == '"' or c == "'") and tag:
            quote = not quote
        elif not tag:
            out = out + c
    # return stripped string
    return out

def generate_wikipedia_AAM():
        html_contents = {}
        # dictionary to hold document frequency of each term in corpus
        terms_df = {}
        # list of all terms
        term_list = []

        # read artist names from file
        artists = Wikipedia_Fetcher.read_file(Wikipedia_Fetcher.ARTISTS_FILE)   # using functions and parameters defined in Wikipedia_Fetcher.py

        helper.ensure_dir(WIKIPEDIA_OUTPUT)

        # for all artists
        for i in range(0, len(artists)):
            # construct file name to fetched HTML page for current artist, depending on parameter settings in Wikipedia_Fetcher.py
            if Wikipedia_Fetcher.USE_INDEX_IN_OUTPUT_FILE:
                html_fn = Wikipedia_Fetcher.OUTPUT_DIRECTORY + "/" + str(i) + ".html"     # target file name
            elif not Wikipedia_Fetcher.USE_INDEX_IN_OUTPUT_FILE:
                html_fn = Wikipedia_Fetcher.OUTPUT_DIRECTORY + "/" + urllib.quote(artists[i]) + ".html"     # target file name

            # Load fetched HTML content if target file exists
            if os.path.exists(html_fn):
                # Read entire file
                html_content = open(html_fn, 'r').read()

                # Next we perform some text processing:
                # Strip content off HTML tags
                content_tags_removed = remove_html_markup(html_content)
                # Perform case-folding, i.e., convert to lower case
                content_casefolded = content_tags_removed.lower()
                # Tokenize stripped content at white space characters
                tokens = content_casefolded.split()
                # Remove all tokens containing non-alphanumeric characters; using a simple lambda function (i.e., anonymous function, can be used as parameter to other function)
                tokens_filtered = filter(lambda t: t.isalnum(), tokens)
                # Remove words in the stop word list
                tokens_filtered_stopped = filter(lambda t: t not in STOP_WORDS, tokens_filtered)
                # Store remaining tokens of current artist in dictionary for further processing
                html_contents[i] = tokens_filtered_stopped
                print "File " + html_fn + " --- total tokens: " + str(len(tokens)) + "; after filtering and stopping: " + str(len(tokens_filtered_stopped))
            else:           # Inform user if target file does not exist
                print "Target file " + html_fn + " does not exist!"

        print html_contents
        # Start computing term weights, in particular, document frequencies and term frequencies.

        # Iterate over all (key, value) tuples from dictionary just created to determine document frequency (DF) of all terms
        for aid, terms in html_contents.items():
            # convert list of terms to set of terms ("uniquify" words for each artist/document)
            for t in set(terms):                         # and iterate over all terms in this set
                # update number of artists/documents in which current term t occurs
                if t not in terms_df:
                    terms_df[t] = 1
                else:
                    terms_df[t] += 1

        # Compute number of artists/documents and terms
        no_artists = len(html_contents.items())
        no_terms = len(terms_df)
        print "Number of artists in corpus: " + str(no_artists)
        print "Number of terms in corpus: " + str(no_terms)

        # You may want (or need) to perform some kind of dimensionality reduction here, e.g., filtering all terms
        # with a very small document frequency.
        # ...


        # Dictionary is unordered, so we store all terms in a list to fix their order, before computing the TF-IDF matrix
        for t in terms_df.keys():
            term_list.append(t)


        # Create IDF vector using logarithmic IDF formulation
        idf = np.zeros(no_terms, dtype=np.float32)
        for i in range(0, no_terms):
            idf[i] = np.log(no_artists / terms_df[term_list[i]])
    #        print term_list[i] + ": " + str(idf[i])

        # Initialize matrix to hold term frequencies (and eventually TF-IDF weights) for all artists for which we fetched HTML content
        tfidf = np.zeros(shape=(no_artists, no_terms), dtype=np.float32)

        # Iterate over all (artist, terms) tuples to determine all term frequencies TF_{artist,term}
        terms_index_lookup = {}         # lookup table for indices (for higher efficiency)
        for a_idx, terms in html_contents.items():
            print "Computing term weights for artist " + str(a_idx)
            # You may want (or need) to make the following more efficient.
            for t in terms:                     # iterate over all terms of current artist
                if t in terms_index_lookup:
                    t_idx = terms_index_lookup[t]
                else:
                    t_idx = term_list.index(t)      # get index of term t in (ordered) list of terms
                    terms_index_lookup[t] = t_idx
                tfidf[a_idx, t_idx] += 1        # increase TF value for every encounter of a term t within a document of the current artist

        # Replace TF values in tfidf by TF-IDF values:
        # copy and reshape IDF vector and point-wise multiply it with the TF values
        tfidf = np.log1p(tfidf) * np.tile(idf, no_artists).reshape(no_artists, no_terms)

        # Storing TF-IDF weights and term list
        print "Saving TF-IDF matrix to " + WIKIPEDIA_TFIDFS + "."
        np.savetxt(WIKIPEDIA_TFIDFS, tfidf, fmt='%0.6f', delimiter='\t', newline='\n')

        print "Saving term list to " + WIKIPEDIA_TERMS + "."
        with open(WIKIPEDIA_TERMS, 'w') as f:
            for t in term_list:
                f.write(t + "\n")

        # Computing cosine similarities and store them
    #    print "Computing cosine similarities between artists."
        # Initialize similarity matrix
        sims = np.zeros(shape=(no_artists, no_artists), dtype=np.float32)
        # Compute pairwise similarities between artists
        for i in range(0, no_artists):
            print "Computing similarities for artist " + str(i)
            for j in range(i, no_artists):
                cossim = 1.0 - scidist.cosine(tfidf[i], tfidf[j])

                # If either TF-IDF vector (of i or j) only contains zeros, cosine similarity is not defined (NaN: not a number).
                # In this case, similarity between i and j is set to zero (or left at zero, in our case).
                if not np.isnan(cossim):
                    sims[i,j] = cossim
                    sims[j,i] = cossim

        print "Saving cosine similarities to " + WIKIPEDIA_AAM + "."
        np.savetxt(WIKIPEDIA_AAM, sims, fmt='%0.6f', delimiter='\t', newline='\n')

def generate_musixmatch_AAM():
    ps = PorterStemmer()
    lyrics_contents  = {}
    artist_id_object = {}
    terms_df         = {}
    term_list        = []

    musixmatch_artists = mf.read_txt(mf.GENERATED_ARTISTS_FILE)

    counter = 0
    artist_counter = 0

    # filtering words
    for artist_id, artist_name in musixmatch_artists.items():
        if (counter < 10):
            file = mf.OUTPUT_DIR_MUSIXMATCH_JSON + str(artist_id) + '.json'

            try:
                with open(file, 'r') as f:
                    data  = json.load(f)      # create reader
                    data_by_artist = data[artist_id]
                    lyrics_content = ''

                    for string in data_by_artist:
                        # remove all non english
                        try:
                            lang = detect(string)

                            if lang == 'en':
                                lyrics_content += re.sub(r'\*.*\*(\s|\S)*$', '', string)
                        except:
                            continue;

                    # remove dots
                    content_no_dots = re.sub(r'\.', ' ', lyrics_content)

                    # remove numbers
                    content_no_numbers = re.sub(r'[0-9]+', ' ', content_no_dots)

                    # Perform case-folding, i.e., convert to lower case
                    content_casefolded = content_no_numbers.lower()

                    # Tokenize stripped content at white space characters
                    tokens = content_casefolded.split()

                    # Remove all tokens containing non-alphanumeric characters; using a simple lambda function (i.e., anonymous function, can be used as parameter to other function)
                    tokens_filtered = filter(lambda t: t.isalnum(), tokens)

                    # Remove words in the stop word list
                    tokens_filtered_stopped = filter(lambda t: t not in STOP_WORDS, tokens_filtered)

                    tokens_stammed = []

                    for w in tokens_filtered_stopped:
                        tokens_stammed.append(ps.stem(w))

                    if len(tokens_stammed) > 0:
                        artist_id_object[artist_counter] = artist_id
                        lyrics_contents[artist_counter]  = tokens_stammed
                        artist_counter += 1

            except:
                print 'File ' + file + ' not found'

        counter += 1

    # get terms list
    # Iterate over all (key, value) tuples from dictionary just created to determine document frequency (DF) of all terms
    for aid, terms in lyrics_contents.items():
        # convert list of terms to set of terms ("uniquify" words for each artist/document)
        for t in set(terms):                         # and iterate over all terms in this set
            # update number of artists/documents in which current term t occurs
            if t not in terms_df:
                terms_df[t] = 1
            else:
                terms_df[t] += 1

    # Compute number of artists/documents and terms
    no_artists = len(lyrics_contents.items())
    no_terms = len(terms_df)
    print "Number of artists in corpus: " + str(no_artists)
    print "Number of terms in corpus: " + str(no_terms)

    # You may want (or need) to perform some kind of dimensionality reduction here, e.g., filtering all terms
    # with a very small document frequency.
    # ...

    # Dictionary is unordered, so we store all terms in a list to fix their order, before computing the TF-IDF matrix
    for t in terms_df.keys():
        term_list.append(t)

    # Create IDF vector using logarithmic IDF formulation
    idf = np.zeros(no_terms, dtype=np.float32)
    for i in range(0, no_terms):
        idf[i] = np.log(no_artists / terms_df[term_list[i]])

    # Initialize matrix to hold term frequencies (and eventually TF-IDF weights) for all artists for which we fetched HTML content
    tfidf = np.zeros(shape=(no_artists, no_terms), dtype=np.float32)

    # without stemming
    # artists max 20
    # 16 artists
    # 1826 words
    # print tfidf

    # Iterate over all (artist, terms) tuples to determine all term frequencies TF_{artist,term}
    terms_index_lookup = {}         # lookup table for indices (for higher efficiency)
    for a_idx, terms in lyrics_contents.items():
        print "Computing term weights for artist " + str(a_idx)
        # You may want (or need) to make the following more efficient.
        for t in terms:                     # iterate over all terms of current artist
            if t in terms_index_lookup:
                t_idx = terms_index_lookup[t]
            else:
                t_idx = term_list.index(t)      # get index of term t in (ordered) list of terms
                terms_index_lookup[t] = t_idx
            tfidf[a_idx, t_idx] += 1        # increase TF value for every encounter of a term t within a document of the current artist

    # Replace TF values in tfidf by TF-IDF values:
    # copy and reshape IDF vector and point-wise multiply it with the TF values
    tfidf = np.log1p(tfidf) * np.tile(idf, no_artists).reshape(no_artists, no_terms)

    # Storing TF-IDF weights and term list
    print "Saving TF-IDF matrix to " + MUSIXMATCH_TFIDFS + "."
    np.savetxt(MUSIXMATCH_TFIDFS, tfidf, fmt='%0.6f', delimiter='\t', newline='\n')

    print "Saving term list to " + MUSIXMATCH_TERMS + "."
    with open(MUSIXMATCH_TERMS, 'w') as f:
        for t in term_list:
            f.write(t + "\n")

    # Computing cosine similarities and store them
#    print "Computing cosine similarities between artists."
    # Initialize similarity matrix
    sims = np.zeros(shape=(no_artists, no_artists), dtype=np.float32)
    # Compute pairwise similarities between artists
    for i in range(0, no_artists):
        print "Computing similarities for artist " + str(i)
        for j in range(i, no_artists):
            cossim = 1.0 - scidist.cosine(tfidf[i], tfidf[j])

            # If either TF-IDF vector (of i or j) only contains zeros, cosine similarity is not defined (NaN: not a number).
            # In this case, similarity between i and j is set to zero (or left at zero, in our case).
            if not np.isnan(cossim):
                sims[i,j] = cossim
                sims[j,i] = cossim

    print "Saving cosine similarities to " + MUSIXMATCH_AAM + "."
    np.savetxt(WIKIPEDIA_AAM, sims, fmt='%0.6f', delimiter='\t', newline='\n')

    print "Saving AAM artist id to " + MUSIXMATCH_AAM_ARTIST_ID + "."

    mf.save_txt(artist_id_object, MUSIXMATCH_AAM_ARTIST_ID, MUSIXMATCH_OUTPUT)
# /generate_musixmatch_AAM

def generate_musixmatch_artist_artists():
    mm_artists = mf.read_txt(MUSIXMATCH_ARTISTS_ID)
    aam_artists = mf.read_txt(MUSIXMATCH_OUTPUT + MUSIXMATCH_AAM_ARTIST_ID)

    aam_artists_list = []

    for aam_id, aam_mm_id in aam_artists.items():
        for mm_aid, mm_artist_name in mm_artists.items():
            if aam_mm_id == mm_aid:
                aam_artists_list.append(mm_artist_name)

    # mf.save_txt(aam_artists_list, 'artists.txt', MUSIXMATCH_OUTPUT)
# /generate_musixmatch_artist_artists


# Main program
if __name__ == '__main__':
    # dictionary to hold tokenized HTML content of each artist
    #generate_wikipedia_AAM()
    #generate_musixmatch_AAM()
    generate_musixmatch_artist_artists()