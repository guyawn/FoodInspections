from gensim.test.utils import datapath
from gensim import utils
from gensim import models


# Class to load the North Carolina restaurant text
class CorpusNC(object):

    def __iter__(self):

        corpus_path = datapath(
            'C:\\Users\\gsene\\OneDrive\\Desktop\\Coursework\\Duke\\Fall 2019\\ECE590-13 NLP\\FoodInspections\\Data\\EmbeddingModels\\reviews_nc.cor'
        )
        for line in open(corpus_path):
            yield utils.simple_preprocess(line)


if __name__ == "__main__":

    sizes = [50, 100, 200, 300]
    sgs = ["cbow", "skip"]

    sentences_nc = CorpusNC()

    for size in sizes:
        for sg in sgs:

            save_name = "Data\\EmbeddingModels\\" + \
                        "W2V_NC_" + sg + \
                        "Size_" + str(size) + \
                        ".model"

            print("Making " + save_name)

            if sg == "cbow":
                current_model = models.Word2Vec(sentences=sentences_nc,
                                                size=size,
                                                sg=0)
                current_model.save(save_name)

            if sg == "skip":
                current_model = models.Word2Vec(sentences=sentences_nc,
                                                size=size,
                                                sg=1)
                current_model.save(save_name)
