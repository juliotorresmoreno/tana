from knowledge.library import Library
from knowledge.elasticsearch import ElasticSearchLibrary

def make_library(index_name: str) -> Library:
    lib = ElasticSearchLibrary(index_name)
    return lib
