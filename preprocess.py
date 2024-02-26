import re

class Preprocessing(object):
    stopwords_list = []
    def __init__(self):
        with open('stopwords', 'r', encoding='utf-8') as f:
            self.stopwords_list = f.read().splitlines()
        pass

    def newline(self, input_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            contents = f.read()
        contents = contents.replace('. ', '\n')
        contents = contents.replace(', ', ' ')
        contents = contents.replace('/', ' ')
        contents = contents.replace('“', ' ')
        contents = contents.replace('”', ' ')
        contents = contents.replace('!', ' ')
        contents = contents.replace('’', '\'')
        with open('corpus', 'w', encoding='utf-8') as f:
            f.write(contents)
    
    def stripempty(self, input_file):
        
        def strip(line):
            newline = line.lstrip()
            newline = newline.rstrip()
            newline = newline.rstrip('.')
            newline = re.sub("[\(\[].*?[\)\]]", "", newline)
            return newline
        
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            newlines = [strip(line) for line in lines]
        with open(input_file, 'w', encoding='utf-8') as f:
            for line in newlines:
                if re.match(r'^\s*$', line):
                    continue
                else:
                    f.write(line+'\n')

    def clean_stopwords(self, input_file):
        def clean_line(input_line):
            line = input_line.lower()
            words = line.split()
            clean_words = [word for word in words if (word not in self.stopwords_list)]
            # print(" ".join(clean_words))
            return " ".join(clean_words)

        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            newlines = [clean_line(line) for line in lines]
        with open(input_file, 'w', encoding='utf-8') as f:
            for line in newlines:
                    f.write(line+'\n')

if __name__ == '__main__':
    preprocess = Preprocessing()
    input = 'corpus_backup'
    preprocess.newline(input)
    preprocess.stripempty('corpus')
    preprocess.clean_stopwords('corpus')
