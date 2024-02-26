import gensim

if __name__ == '__main__':
    input_model_path = 'model.mdl'
    model = gensim.models.Word2Vec.load(input_model_path)
    # print(list(model.wv.index_to_key))
    print(model.wv.most_similar(positive="kema", topn=5))