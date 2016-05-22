import logging
import os
import re
import sys
import html.parser
import itertools
from string import punctuation
from pickle import Pickler, load

from gensim.parsing.preprocessing import STOPWORDS
from gensim.corpora.dictionary import Dictionary
from gensim.corpora import MmCorpus
from gensim.models.tfidfmodel import TfidfModel
from gensim.similarities import SparseMatrixSimilarity
from gensim.utils import simple_preprocess

from textblob import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag

from potatobot import Answer, Followup, PotatoBot

html_parser = html.parser.HTMLParser()
TAG_RE = re.compile(r'<[^><]+>')
CODE_RE = re.compile(r'<pre>.*?</pre>', re.DOTALL)
lemmatizer = WordNetLemmatizer()

CONTAINER_DIR = 'containers/'
CORPUS_FILE = '-eecs281corpus.mm'
DICTIONARY_FILE = '-eecs281.dict'
ID_MAP_FILE = '-281corpus_id_map.pickle'

TFIDF_THRESHOLD = .375
NUM_MIN_TERMS = 5
SIM_LIMIT = 5

CPP_KEYWORDS = {
    'auto',
    'bool',
    'break',
    'case',
    'catch',
    'char',
    'class',
    'const',
    'cast',
    'continue',
    'delete',
    'do',
    'double',
    'else',
    'enum',
    'struct',
    'extern',
    'false',
    'for',
    'float',
    'each',
    'endif',
    'friend',
    'goto',
    'if',
    'ifdef',
    'include',
    'int',
    'namespace',
    'new',
    'nullptr',
    'operator',
    'private',
    'protected',
    'public',
    'return',
    'short',
    'signed',
    'sizeof',
    'static',
    'switch',
    'std'
    'this',
    'throw',
    'true',
    'try',
    'typedef',
    'union',
    'unsigned',
    'using',
    'virtual',
    'void',
    'while'}

CONTRACTIONS = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it had",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there had",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'alls": "you alls",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had",
    "you'd've": "you would have",
    "you'll": "you you will",
    "you'll've": "you you will have",
    "you're": "you are",
    "you've": "you have"}

CONTRACTIONS_RE = re.compile('(%s)' % '|'.join(CONTRACTIONS.keys()))


def expand_contractions(text):
    def replace(match):
        return CONTRACTIONS[match.group(0)]
    return CONTRACTIONS_RE.sub(replace, text)


def die(message):
    sys.stderr.write("{}\n".format(message))
    sys.exit(1)


def get_bot():
    config_sources = {
        "email": "PBOT_EMAIL",
        "password": "PBOT_PASSWORD",
        "class_code": "PBOT_CLASS_CODE",
    }

    config = {}
    for var, source in config_sources.items():
        try:
            config[var] = os.environ[source]
        except KeyError:
            die("`{}` must be set in the environment.".format(
                source
            ))

    return PotatoBot.create_bot(**config)


def has_uniqname(display_name):
    """Whether or not the given user has their uniqname in their Piazza display
    name, as per @6.

    display_name: Their display name on Piazza.

    """
    if "Piazza Team" in display_name:
        return True

    username_regex = re.compile(r"""
        \(
            [^) ]{1,8} # Their uniqname, in parentheses. There's a maximum
                      # of eight characters in a uniqname.
        \)
    """, re.VERBOSE)

    # Search for their uniqname anywhere in their Piazza display name.
    match = username_regex.search(display_name)
    return match is not None


def test_has_uniqname():
    assert not has_uniqname("Waleed Khan")
    assert has_uniqname("Waleed Khan (wkhan)")
    assert has_uniqname("(wkhan)")
    assert not has_uniqname("(wkhan")
    assert not has_uniqname("( wkhan )")


def test_has_uniqname_ignores_special_usernames():
    assert has_uniqname("Piazza Team")


def is_bad_compiler_error_post(post_text):
    """Detect whether the student is trying to post a message about a compiler
    error, but hasn't actually given us the compiler error."""
    post_text = post_text.lower()
    post_text = post_text.replace("&#39;", "'")

    if "#compiler-error" in post_text:
        return True

    compile_words = ["compile", "compiling", "compilation"]
    if not any(i in post_text for i in compile_words):
        return False

    # Decide whether it's a post about an error.
    about_error = ["error", "not work", "doesn't work", "won't work",
                   "not compil", "complain"]
    if not any(i in post_text for i in about_error):
        return False

    # Now, see if they've actually given us anything to work with.
    provided_error = ["<code", "<pre", "<img", ": error:"]
    if any(i in post_text for i in provided_error):
        return False

    return True


def test_is_bad_compiler_error_post():
    assert is_bad_compiler_error_post("regular post #compiler-error")
    assert is_bad_compiler_error_post("Compile error")
    assert is_bad_compiler_error_post("compile doesn't work")
    assert is_bad_compiler_error_post("compile doesn&#39;t work")
    assert is_bad_compiler_error_post("not working compiler")
    assert is_bad_compiler_error_post("not compiling")
    assert is_bad_compiler_error_post("compiler complaining")
    assert not is_bad_compiler_error_post("compilers are great")
    assert not is_bad_compiler_error_post("my code isn't working")
    assert not is_bad_compiler_error_post("not working compile message <pre>")
    assert not is_bad_compiler_error_post("not working compile message <code>")
    assert not is_bad_compiler_error_post("error compile message <img src>")
    assert not is_bad_compiler_error_post("not compiling 90: error:")


def cant_valgrind(post_text):
    """Whether or not the student is trying to debug a segfault but hasn't
    tried valgrind."""
    post_text = post_text.lower()

    if "#valgrind" in post_text:
        return True

    segfault_error = ["segv", "sigsev", "sig sev",
                      "segfault", "seg fault",
                      "segmentation"]
    if not any(i in post_text for i in segfault_error):
        return False

    if "valgrind" in post_text:
        return False

    return True


def test_cant_valgrind():
    assert cant_valgrind("regular post #valgrind")
    assert cant_valgrind("SIG SEV but don't know why")
    assert cant_valgrind("SIGSEV but don't know why")
    assert cant_valgrind("SIGSEGV but don't know why")
    assert cant_valgrind("my program segfaults")
    assert cant_valgrind("my program seg faults")
    assert cant_valgrind("I have a segmentation issue")
    assert not cant_valgrind("segfault valgrind doesn't help")
    assert not cant_valgrind("my program sucks")


def main():
    logging.basicConfig(level=logging.DEBUG)
    bot = get_bot()

    @bot.handle_post
    def demand_uniqname(post_info):
        if not has_uniqname(post_info.username):
            return Answer("""
<p>Hi! It looks like you don't have your uniqname in your display name, as per
@6. Please add it so that we can look you up on the autograder quickly.</p>
""")

    @bot.handle_post
    def complain_about_compiler_errors(post_info):
        if is_bad_compiler_error_post(post_info.text):
            return Answer("""
<p>Hi! It looks like you have a compiler error, but you didn't tell us what the
error was! (Or if you did, you didn't paste it into a code block so that we
could read it easily.) We'll need to see the <em>full</em> compile error
output, so please add it.</p>
<p></p>
<p>Here are some guidelines you will need to follow if you are to receive help:

<ul>
  <li>Paste the <strong>FULL COMPILER ERROR</strong> from g++. Do not omit any
  part.</li>
  <li>Do not paste compiler errors from Visual Studio or Xcode, as they are
  usually one line long and not informative. Please compile your code under
  g++, as it provides infinitely more helpful compiler error messages.</li>
  <li>Do not post part of the compiler error from g++ and say that the rest is
  similar to it. That is not helpful and someone on staff will scowl at you.
  Paste the entire compiler error output, even the parts you think are not
  important.</li>
  <li>Even if you think your question can be answered without looking at the
  compiler error message, please post it anyways.</li>
</ul>

<p></p>
<p>If you don't have a compiler error, sorry about that! I'm just a potato; I
can't read very well.</p>
""")

    @bot.handle_post
    def learn_to_valgrind_please(post_info):
        if cant_valgrind(post_info.text):
            return Answer("""
<p>It looks like you're having an issue with a segfault! Have you run your code
under valgrind? If you don't know how to use valgrind, read this: <a
href="http://maintainablecode.logdown.com/posts/245425-valgrind-is-not-a-leak-checker">Valgrind
is not a leak-checker</a>.</p>
<p></p>
<p>Having no memory leaks does not indicate that your program is safe. Read the
above article.</p>
<p></p>
<p>Once you've valgrinded your code, post the full valgrind output and the
relevant lines of code. (Make sure that you compile <code>make debug</code> so
that valgrind shows you line numbers.) <strong>We will not answer your question
until you post valgrind output</strong>.</p>
<p></p>
<p>If valgrind doesn't show anything, that probably means you need better test
cases!</p>
""")

    @bot.handle_post
    def check_for_duplicate_posts_tfidf(post_info):
        """
        use tfidf to generate a list of posts that are
        similar to the post provided in post_info

        For more information:

        https://radimrehurek.com/gensim/models/tfidfmodel.html
        http://radimrehurek.com/gensim/corpora/dictionary.html
        http://radimrehurek.com/gensim/tutorial.html
        """
        sim_list = set()
        for folder in post_info.folders:
            dictionary, corpus, corpus_id_to_true_id = read_containers(folder)
            terms = get_terms(post_info.text)
            if (post_info.status != "private" and
                    post_info.id not in corpus_id_to_true_id[-50:]):
                update_containers_with_terms(
                    terms,
                    corpus,
                    dictionary,
                    corpus_id_to_true_id,
                    post_info.id)
                save_containers(
                    folder,
                    corpus,
                    dictionary,
                    corpus_id_to_true_id)

            tfidf = TfidfModel(corpus, id2word=dictionary)
            query_vec = tfidf[dictionary.doc2bow(terms)]
            sim_index = SparseMatrixSimilarity(
                tfidf[corpus],
                num_features=len(dictionary),
                num_best=SIM_LIMIT)

            sim_list |= {corpus_id_to_true_id[sim[0]]
                         for sim in sim_index[query_vec]
                         if sim[1] > TFIDF_THRESHOLD
                         and corpus_id_to_true_id[sim[0]] != post_info.id}
        answers = ", ".join("@{}".format(x) for x in sim_list)
        if answers:
            return Followup("""
<p>Hi! It looks like this question has been asked before or there is a related post.
Please look at these posts: {}</p>
<p></p>
<p>If you found your answer in one of the above, please specify which one answered your question.</p>
<p><sub>These posts suggested using TFIDF</sub></p>
""".format(answers))

    bot.run_forever()


def read_containers(folder):
    try:
        dictionary = Dictionary.load(CONTAINER_DIR + folder + DICTIONARY_FILE)
        corpus = list(MmCorpus(CONTAINER_DIR + folder + CORPUS_FILE))
        with open(CONTAINER_DIR + folder + ID_MAP_FILE, 'rb') as id_map_file:
            corpus_id_to_true_id = load(id_map_file)
        return dictionary, corpus, corpus_id_to_true_id
    except:
        return Dictionary(), [], []


def update_containers_with_terms(terms, corpus, dictionary, id_map, postid):
    if len(terms) >= NUM_MIN_TERMS:
        corpus.append(dictionary.doc2bow(terms, allow_update=True))
        id_map.append(postid)


def save_containers(folder, corpus, dictionary, id_map):
    MmCorpus.serialize(CONTAINER_DIR + folder + CORPUS_FILE, corpus)
    dictionary.save(CONTAINER_DIR + folder + DICTIONARY_FILE)
    with open(CONTAINER_DIR + folder + ID_MAP_FILE, 'wb') as picklefile:
        Pickler(picklefile).dump(id_map)


def get_terms(content):
    text = html_parser.unescape(content)

    code = [set(simple_preprocess(x[5:-6].lower())) -
            CPP_KEYWORDS - STOPWORDS for x in CODE_RE.findall(text)]
    code = set(itertools.chain(*code))

    text = expand_contractions(TAG_RE.sub('', CODE_RE.sub('', text)).lower())
    text = {lemmatizer.lemmatize(word, get_wordnet_pos(tag))
            for (word, tag) in pos_tag(word_tokenize(text))
            if word not in STOPWORDS and word not in punctuation}
    text = {item.strip(punctuation)
            for item in text if item.strip(punctuation)}
    return text | code


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

if __name__ == "__main__":
    main()
