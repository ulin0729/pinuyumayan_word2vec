from gensim.models import Word2Vec
import multiprocessing

def get_sentence_list(input_file):
    ret = []
    with open(input_file, 'r', encoding='utf-8') as f:
        contents = f.readlines()
    for lines in contents:
        words = lines.split()
        ret.append(words)
    return ret

if __name__ == '__main__':
    input_path = 'corpus'
    output_model_path = 'model.mdl'
    sentences_list = get_sentence_list(input_path)
    cores = multiprocessing.cpu_count()
    model = Word2Vec(min_count=3, window=5, vector_size=300, workers=cores-1, max_vocab_size=None)
    model.build_vocab(sentences_list)
    model.train(sentences_list, total_examples = model.corpus_count, epochs = 200)
    model.save(output_model_path)