from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


def make_tfidf(max_features=5000, ngram_range=(1, 2)):
    return TfidfVectorizer(max_features=max_features, ngram_range=ngram_range,
                           sublinear_tf=True, min_df=2, max_df=0.9)


def build_pipeline(classifier, **tfidf_kwargs):
    return Pipeline([("tfidf", make_tfidf(**tfidf_kwargs)), ("clf", classifier)])
