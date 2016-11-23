__author__ = 'jpeer'

###########
# IMPORTS #
###########
import re
import os
import csv
import json
import helper # helper.py
import operator
from langdetect import detect
from nltk.stem import PorterStemmer

####################
# GLOBAL VARIABLES #
####################
ROOT        = "../../Homework2/Task1/output/"
OUTPUT      = './output/stats/'
ARTIST_FILE = ROOT + 'artists.txt'

WIKIPEDIA        = ROOT + "wikipedia/"
MUSIXMATCH       = ROOT + "musixmatch/"
MUSIXMATCH_FILES = MUSIXMATCH + "lyrics_json/"
WIKIPEDIA_FILES  = WIKIPEDIA + "crawls_wikipedia/"

GENERATED_ARTISTS_FILE       = MUSIXMATCH + 'artist_ids_unlimited.txt'
GENERATED_ALBUM_IDS_FILE     = MUSIXMATCH + 'album_ids_unlimited.txt'
GENERATED_ALBUM_IDS_FILE_401 = MUSIXMATCH + 'album_tracks_unlimited.txt'
GENERATED_TRACKS_FILE        = MUSIXMATCH + 'album_tracks_unlimited.txt'

VERBOSE     = True
MAX_ARTISTS = 3000

# Stop words used by Google
STOP_WORDS = [
    "a", "able", "about", "above", "abroad", "according", "accordingly", "across", "actually", "adj", "after", "afterwards", "again", "against", "ago", "ahead", "ain't", "all",
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
    "yourself", "yourselves", "you've", "z", "zero", "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"
]

def read_txt(filename, multiple_values=False):
    file_contents = {}

    with open(filename, 'r') as f:
        reader  = csv.reader(f, delimiter='\t')      # create reader
        headers = reader.next()                     # skip header

        for row in reader:
            key   = row[0]
            value = row[1]

            if multiple_values:
                try:
                    file_contents[key].append(value)
                except:
                    file_contents[key] = []
                    file_contents[key].append(value)
            else:
                file_contents[key] = value

    return file_contents
# /read_txt

def count_mm_terms():
    """
    Counts all terms from the lyrics of MAX_ARTISTS artists

    :return: an dictionary with meta information
    """
    ps = PorterStemmer()
    languages       = {}
    lyrics_contents = {}
    terms_df        = {}
    term_list       = []
    total_string    = ''
    found_artists   = 0
    total_songs     = 0

    musixmatch_artists = read_txt(GENERATED_ARTISTS_FILE)
    artists_file = helper.read_csv(ARTIST_FILE)

    ###########################
    ## keep artist structure ##
    ###########################

    # iterate over the same artist file and check
    # if the values are in the same order
    # so the later generated AAM is still in the same order
    for index, artist_name in enumerate(artists_file, start = 0):
        # make it short for debugging
        if VERBOSE:
            print 'Get lyrics of ' + artist_name + ' [' + str(index + 1) + ' of ' + str(len(artists_file)) + ']'

        if index < len(artists_file):
            for artist_mm_id, artist_mm_name in musixmatch_artists.items():
                # if the name is in the musixmatch array
                # to checking it is still in the same order
                if artist_name == artist_mm_name:
                    # check the lyrics and sort everything
                    should_translate = True
                    filename = MUSIXMATCH_FILES + str(artist_mm_id) + '.json'

                    try:
                        with open(filename, 'r') as f:
                            data  = json.load(f)      # create reader
                            data_by_artist = data[artist_mm_id]
                            lyrics_content = ''
                            lyrics_translated = {}
                            lyrics_translated[artist_mm_id] = []
                            lang_global = False
                            found_artist = False

                            for string in data_by_artist:
                                # remove all non english

                                if len(string) != 0:
                                    found_artist = True
                                    total_songs += 1

                                try:
                                    lyrics = re.sub(r'\*.*\*(\s|\S)*$', '', string)
                                    lang = detect(lyrics)

                                    try:
                                        languages[lang] += 1
                                    except:
                                        languages[lang] = 1

                                    # translate non-english strings
                                    if lang != 'en':
                                        total_string += lyrics

                                        translated_string = ''

                                    else:
                                        translated_string = lyrics

                                    lyrics_translated[artist_mm_id].append(translated_string)
                                    lyrics_content += translated_string
                                except Exception, e:
                                    continue

                            if found_artist:
                                found_artists += 1

                            #####################################
                            ## sorting | stamming | stopwords ##
                            #####################################

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

                            tokens_stemmed = []

                            for w in tokens_filtered_stopped:
                                tokens_stemmed.append(ps.stem(w))

                            lyrics_contents[index] = tokens_stemmed

                    except Exception, e:
                        print e
                        print 'File ' + filename + ' not found'


    # iterate over the max artists and check
    for index in range(0, MAX_ARTISTS):
        try:
            if (lyrics_contents[index]):
                continue;
        except:
            lyrics_contents[index] = ''

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

    terms_df_dict = terms_df
    terms_df = sorted(terms_df.items(), key=operator.itemgetter(1), reverse=True)

    len_all = len(terms_df_dict.items())
    len_limited = len(dict((k, v) for k, v in terms_df_dict.iteritems() if v != 1))

    stats = {}
    stats['len_all'] = len_all
    stats['len_limited'] = len_limited
    stats['best_five'] = terms_df[:5]
    stats['all'] = terms_df
    stats['len_to_translate_char'] = len(total_string)
    stats['lang_stats'] = languages
    stats['found_artists'] = found_artists
    stats['total_songs'] = total_songs

    return stats
# /count_mm_terms

# A simple function to remove HTML tags from a string.
# You can of course also use some fancy library. In particular, lxml (http://lxml.de/) seems a simple and good solution; also for getting rid of javascript.
def remove_html_markup(s):
    """
    straight copied from WebSimilarity.py ...
    Regex would be better, but yeah :)
    r"<.*?>"g would probably do that job too IF all new lines and white spaces are removed before r"\n+\s*"

    :param s: the string to remove the html markup
    """
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
# /remove_html_markup

def count_wiki_terms():
    """
    Counts all terms from wikipedia fetches

    :return: an dictionary with meta information
    """
    ps            = PorterStemmer()
    html_contents = {}
    terms_df      = {}
    term_list     = []
    found_artists = 0

    # read artist names from file
    artists = helper.read_csv(ARTIST_FILE)   # using functions and parameters defined in o1_Wikipedia_Fetcher.py

    # start with one because UAM is false
    # there is no header in the artist file
    # so skip Lil Wayne, that's just Lil' (Bit) Wayne - he appears later again
    for i in range(1, len(artists)):
        # construct file name to fetched HTML page for current artist, depending on parameter settings in Wikipedia_Fetcher.py
        html_fn = WIKIPEDIA_FILES + "/" + str(i) + ".html"     # target file name

        # Load fetched HTML content if target file exists
        if os.path.exists(html_fn):
            # Read entire file
            html_content = open(html_fn, 'r').read()

            if len(html_content) == 0:
                found_artists += 1

            # Next we perform some text processing:
            # Strip content off HTML tags
            content_tags_removed = remove_html_markup(html_content)
            # remove numbers
            content_no_numbers = re.sub(r'[0-9]+', ' ', content_tags_removed)
            # Perform case-folding, i.e., convert to lower case
            content_casefolded = content_no_numbers.lower()
            # remove words with wiki in it
            content_no_specific_words = re.sub(r'[\w]*wiki|article|pedia|privacy|policy[\w]*', ' ', content_casefolded)
            # Tokenize stripped content at white space characters
            tokens = content_no_specific_words.split()
            # Remove all tokens containing non-alphanumeric characters; using a simple lambda function (i.e., anonymous function, can be used as parameter to other function)
            tokens_filtered = filter(lambda t: t.isalnum(), tokens)
            # Remove words in the stop word list
            tokens_filtered_stopped = filter(lambda t: t not in STOP_WORDS, tokens_filtered)

            tokens_stemmed = []
            # stemm words
            for w in tokens_filtered_stopped:
                tokens_stemmed.append(ps.stem(w))

            # Store remaining tokens of current artist in dictionary for further processing
            html_contents[i] = tokens_stemmed

            print "File " + html_fn + " --- total tokens: " + str(len(tokens)) + "; after filtering and stopping: " + str(len(tokens_filtered_stopped))
        else:           # Inform user if target file does not exist
            print "Target file " + html_fn + " does not exist!"
            found_artists += 1
            html_contents[i] = ''

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

    terms_df_dict = terms_df
    terms_df = sorted(terms_df.items(), key=operator.itemgetter(1), reverse=True)

    len_all = len(terms_df_dict.items())
    len_limited = len(dict((k, v) for k, v in terms_df_dict.iteritems() if v != 1))

    stats = {}
    stats['len_all'] = len_all
    stats['len_limited'] = len_limited
    stats['best_five'] = terms_df[:5]
    stats['all'] = terms_df
    stats['found_artists'] = found_artists

    return stats
# /count_wiki_terms

if __name__ == '__main__':
    loop_me = {}
    #loop_me['wiki'] = count_wiki_terms()
    loop_me['mm']   = count_mm_terms()

    helper.ensure_dir(OUTPUT)

    for key, terms in loop_me.iteritems():
        filename = 'novalue.json'

        if key == 'wiki':
            filename = 'wiki_term_stats.json'

        elif key == 'mm':
            filename = 'mm_term_stats.json'

        content   = json.dumps(terms, indent=4, sort_keys=True)
        json_file = open(OUTPUT + 'mm_term_stats.json', 'w')

        json_file.write(content)
        json_file.close()

