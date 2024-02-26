# DSP 2022 Final project

# 南島語族之 Word2Vec 模型 --- 以西群卑南語為例

B10902128 資工三 曾宥林

## 1. 前言

台灣是一個擁有多元文化的地方，除了目前大多數台灣人所使用的漢語，這片土地上還存有各式各樣的語言，如台語、客語、原住民語等。其中，原住民語由於原住民的人口基數較少，使用原住民語的人也寥寥無幾，加上漢語的強勢語言特性，正面臨著語言消亡的危機。

在這樣的情況下，或許資訊科技可以以某種方式幫助保存並傳承這些語言，例如課堂中所提及的 Computer-Assisted Language Learning，或者是機器翻譯等系統，都可以降低學習語言的門檻，進而提升這些語言傳承下去的機率。

考量到台灣南島語有二十餘種，而且每一種都有各自不同的方言別，再加上語料資料庫獲取不易，要在這種 Low resource 的語言上訓練出良好或甚至堪用的模型並不容易，但是相信眾志終能成城，謹以此專案拋磚引玉，希望能讓更多的人願意投入這塊研究。



## 2. Word2Vec 簡介

機器不是人，沒辦法「望詞生義」，因此我們需要將詞用一種機器看得懂的方式來表述，這樣一來詞才能被送進機器，並讓機器讀懂這個詞所蘊含的資訊。

Word embedding 正是在做這件事情，將詞映射到向量空間上的某一點，如此一來便能用一個 vector 來表示一個詞。其中 word embedding 的輸入為一個詞，輸出為一個向量。

若要用一個 vector 來表達一個詞，要怎麼做呢？

很直覺地，最簡單的方式就是用 one-hot encoding，給每一個詞指定一個位置，這個詞的 vector 就是全部為 0，只有被指定的那個位置為 1。例如：人：`[1,0,0,...,0,0]`、狗：`[0,1,0,...,0,0]`。不過這樣做有兩個明顯的缺點：

1. 假設一個語言有 $n$ 個不同的詞，那麼每個詞就需要 $n$-d vector 來表示。以英語來說，約有接近二十萬個不同的詞，每個詞都用二十萬維 vector 來表示不僅浪費空間，也不利於機器進行運算。

2. 該 vector 沒辦法提供任何額外資訊。以 one-hot encode 的 vector，所提供的資訊量就僅僅只有這個詞是或不是某個詞，沒辦法提供語其他詞之間的關聯性這種有用的額外資訊。

2013 年時，Tomas Mikolov 提出了 Word2Vec 的方法，以神經網路為基礎，並搭配上一些統計模型，得出一個能夠將詞轉為 vector 的方法，而且這個 vector 內還能蘊含相似性的資訊！



## 3. 實驗步驟

### 1. 收集語料

語料的部分是從教育部原住民語朗讀比賽 105 至 111 年度的公告文章節錄下來的，共 45 篇文章，約 15000 個詞。

格式大概如下

```bash
mudawilr mukuwa muzazangiya
“ inta na trau palru za minatray ’azi ta ziya pakalrang mupatraran mudawilr muzazangi, nanta ikavavaaw mu, iniyan za trima’. ” kema merengay na trau. nu mudawilr ta mukuwa muzazangiya mu, pakasaazu zata inyana’u zata kiningeran.
nu kazu ta i Taiwan mu, ulra na trau pakakawang kana palizing mukuwa muzazangiya, ulra na muitras kana otobay, na zidinsiya mukuwa muzangizangiya. nu kazu ta i Taiwan mu, iturus zata anger mukuwa kanta auwayan.
na ma’izang lra tu ’ami na trau mu, kurelrang kana lepun muitras kana yuranbas(遊覽車) mukuwa muzangiya, i Taiwan sazu na marekamelrimelri na zekalr nantu kinakuwakuwa na kivangavangan, zi mau nantu vinilin na kinaawatrawatranan na eman ziya.
...
```

註：卑南語使用英語中的單引號「'」來表示濁喉擦音 /ɦ/。

裡面有一些問題：包含標點符號、括號的註解、句子沒有分開。

因此需要做一些格式化。

### 2. 語料格式化

將標點符號去除，並且把每一句以句號為結尾的的句子分成一行一行的，以及去除掉括號內的東西，處理完後類似這樣：

```tex
mudawilr mukuwa muzazangiya
inta na trau palru za minatray 'azi ta ziya pakalrang mupatraran mudawilr muzazanginanta ikavavaaw muiniyan za trima'
kema merengay na trau
nu mudawilr ta mukuwa muzazangiya mupakasaazu zata inyana'u zata kiningeran
nu kazu ta i Taiwan muulra na trau pakakawang kana palizing mukuwa muzazangiyaulra na muitras kana otobayna zidinsiya mukuwa muzangizangiya
nu kazu ta i Taiwan muiturus zata anger mukuwa kanta auwayan
na ma'izang lra tu 'ami na trau mukurelrang kana lepun muitras kana yuranbas mukuwa muzangiyai Taiwan sazu na marekamelrimelri na zekalr nantu kinakuwakuwa na kivangavanganzi mau nantu vinilin na kinaawatrawatranan na eman ziya
```



### 3. 資料前處理

#### 1. 停用詞 Stopwords

在訓練模型時，有許多的詞實際上並沒有意義，例如英文中的 `[is, are, the, ...]` 等，這些詞是用來標記一些所有格等資訊，其本身並沒有實際意義。而卑南語中也存在這樣的詞，例如格位標記的 `[a, i, za, na, ni, ...]` 等、代名詞的 `[ku, lra, ta, ...]` 等、指示代名詞 `[ini, inia, nani, ...]`。因此要創一個 stopword list，用以排除掉這些訓練時沒有意義的詞。

我的實作中以 python script 進行 stopwords 的篩選。

#### 2. 少見詞

在語料資料中，有些詞出現的次數很少，不足以反映該詞與其他詞之間的關聯性，反而容易得到偏頗的結論，因此可以選擇一定詞頻以下的詞都棄用，如此一來既能增加 trainabiliy，也可以避免掉偏頗的結果。

實作中由 gensim train 的參數 min_count 來完成。

```python
import re

class Preprocessing(object):
    stopwords_list = []
    def __init__(self):
        with open('stopwords', 'r', encoding='utf-8') as f:
            self.stopwords_list = f.read().splitlines()

    def newline(self, input_file):	# 移除標點符號
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
    
    def stripempty(self, input_file):	# 移除行首行末空白及括號
        def strip(line):
            newline = line.lstrip()
            newline = newline.rstrip()
            newline = newline.rstrip('.')
            newline = re.sub("[\(].*?[\)]", "", newline)
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

    def clean_stopwords(self, input_file):	# 將 stopwords 篩掉
        def clean_line(input_line):
            line = input_line.lower()
            words = line.split()
            clean_words = [word for word in words if (word not in self.stopwords_list)]
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

```





### 4. Gensim 訓練

Gensim 是一款 open source 的 python NLP 套件，其中包含了 Word2Vec 的 API 可以使用，將一個 list of list 傳入，其中每一句話都以一個 list 表示，該 list 裡面為該句話的每個詞，將所有資料送入後即可訓練模型。

```python
# Sentences:
# sahar ku u.
# 'inava u ziya?
# ...

input = [['sahar', 'ku', 'u'], ['\'inava', 'u', 'ziya'], ...]
```

```python
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
    model.train(sentences_list, total_examples=model.corpus_count, epochs=200)
    model.save(output_model_path)
```

其中有幾項參數可以做調整：

1. `min_count`：就是上面所說的最低詞頻，若 input 中某一詞出現的頻率低於 `min_count`，則會將其跳過不列入計算。
2. `window`：由於 Word2Vec 是以某個詞的左右區間取 window 進行統計，因此 `window` 就決定了每個詞要往左往右看多少個詞作為一個 window。
3. `vector_size`：此即 word embedding 所輸出的向量之維數，理論上越大越好，能更好地反映各個詞分布的情況，但是越大就需要越多的訓練資料來支撐，且計算量也會上升。
4. `workers`：分配給此模型的運算資源數量。
5. `max_vocab_size`：在詞典很大時，為了避免塞爆記憶體所設，但是由於這次使用的語種詞典很小，算上整個語言也不超過 5000 種不同詞，因此此處設為無上限。
6. `total_examples`：Corpus 中句子的數量。由於 vocab 與 corpus 內的詞不一定相同，因此此數值不一定是 corpus 中的句子數，不過此例中 `build_vocab()` 與 `train()` 的時候所用的資料相同，因此直接用`model.corpus_count` 即可。
7.  `epochs`：訓練要重複幾輪，完整 train 過所有 input list 算做一輪。

還有一些 learning rate 等參數可以調，不過這次我選擇將其他參數都設為預設值。



### 5. 測試

```python
import gensim

if __name__ == '__main__':
    input_model_path = 'model.mdl'
    model = gensim.models.Word2Vec.load(input_model_path)
    print(model.wv.most_similar(positive="lima", topn=5))	# here "lima" is the query word
```

直接將模型載入後去查詢與特定詞最相關的前 5 個詞。



### 6. 範例結果分析

首先，有一個壞消息，由於語料實在太小了，訓練出來的模型無論參數怎麼調都不盡理想，不過仍能從一些結果中推得一些結論。

##### Suwan 狗

```python
[('kaazu', 0.7747364640235901), ('muwasavak', 0.6713159084320068), ('dadalan', 0.6366689205169678), ('dawdawan', 0.6151774525642395), ('kawi', 0.6061995029449463)]
```

kaazu：居住、muwasavak：嫁、dadalan：路、dawdawan：老人家、kawi：樹

狗這個詞只出現在某一篇文章中，並且出現次數為 3，正好是我所設的 threshold 中最少的了，可以看到其實前五個相似詞，與狗都沒有真正太大的關聯。這篇文章的大意是建和部落的遷徙歷史，中間提及老人家看見狗被蛇吃掉等情節，由於語料中就只有這個 context 下有出現 suwan 這個詞，因此計算關聯性後就會得到這樣的結果。

##### Ina 媽媽

```python
[("vuwa'", 0.8101927638053894), ('makawang', 0.6657213568687439), ('keravi', 0.6390901803970337), ('marengay', 0.5891034007072449), ('makiteng', 0.5704137682914734)]
```

vuwa：水果、makawang：走路、keravi：吃晚餐、marengay：說、makiteng：上衣

媽媽這個詞在整個語料中出現了六次，在不同故事中都有出現，這五個相似詞大致上有一些特性，都是家庭活動中會出現的事物，我讓沒看過原始文章的朋友看這個結果，他認為感受的到這些詞之間連結，雖說這並不是一個嚴謹的評量方式，但可以看出在相關性上 Word2Vec 是有能力找出類似詞的。在這些故事中，媽媽分別有切水果、織衣服、與小孩說話等。

##### Lima 五/手

```python
[('liyusan', 0.6974535584449768), ('paretelun', 0.6465038657188416), ('kemakawang', 0.5957969427108765), ('puitras', 0.5924357175827026), ('pakakawang', 0.5858916640281677)]
```

liyusan：星期/禮拜、paretelun：三次、kemakawang：走路、puitras：喊、pakakawang：駕駛

選此詞出來分析是因為這是一個典型的多義詞(甚至所有台灣南島語都是用 lima 代表五跟手)，可以很明顯看出，前兩個相似詞的情境之下，lima 是解作五；而中間兩個比較不明顯，原文故事中其實是手拿銅鑼，邊走路邊敲邊喊來慶祝除草完工，駕駛則很明顯是與手相關，因此後三個相似詞的情境下，lima 解作手。

##### Pinuyumayan 卑南族

```python
[('mikakuwayanan', 0.6246741414070129), ("'inupiz", 0.5943820476531982), ('marepuwalraalrang', 0.5820466876029968), ('muhamut', 0.5523918867111206), ('dawa', 0.537775456905365)]
```

mikakuwayanan：習俗、'inupiz na 'aputr：花環、marepuwalraalrang：組織、muhamut：小米除草完工慶(即上面所說的那個)、dawa：小米

Pinuyumayan 共出現三十次，它屬於專有名詞，基本上與其他詞都不會有太大的相似性，也可以從其分數來看，相較於其他組低了不少。



### 7. Word2Vec 特性

從狗的例子中，就可以看出我前面所說的偏頗是怎麼回事，當一個詞僅出現少數幾次的時候，所反映出的相似性並沒有太大意義。

從 lima 的例子中，可以窺見 Word2Vec 的一個缺點，就是沒辦法根據每個詞所在的上下文進行訓練，基本上產生出來的 vector 無論是在  lima 解作五或解作手的情況下都是一樣的，如此一來當我們將此模型套入到 language model 時，就會喪失掉一部份 context 的資訊。

現代的應用情景中，還有另外幾種 Word embedding 的方式：Glove, ELMo, BERT。

Glove 和 Word2Vec 都是以統計模型來計算，只不過計算方式有所不同，相較於 Word2Vec，Glove 的演算法比較利於平行化，訓練時可能佔有一點優勢，不過最終效能差不多，也一樣有無法考慮 context 的問題。

ELMo 和 BERT 則是在訓練時會考慮 word 的位置，ELMo 使用 LSTM、BERT 使用 Transformer，雖然說這樣做能夠獲得 context 資訊，但是當我們要應用這個 model 時，就需要同一份訓練 word embedding 的 corpus，否則位置的資訊就沒有意義了。不過這兩個相對於上兩個來說，所需要的計算量更大，所需要的語料庫也更多。



### 8. 本次實驗的困難點

最嚴重的其實就是語料庫嚴重不足，眾所周知，原住民語很少人會講，而且因為其口述語言的特性，會寫的人又更少，部落中的耆老多半會講不會寫，只有受過訓練的族語老師或者極少數的有志之士才能流利地說寫原住民語。因此我所能獲取的資料很少，也就造成訓練效果不好，目前也有好幾篇論文是在探討關於這種 low resource 語言的 NLP 模型要如何建造，或許未來能朝此方向去鑽研。

其次，其實本次實驗中有另一個問題，卑南語的語法結構會讓同一個詞產生各種不同的變形，例如 kezeng 是牽的意思，但是當我們在某些時態或語氣之下，要用 kakezeng ；又如 trangis 是哭的意思，但當我要說小孩子在哭時，卻要用 matrangis na alrak，在 trangis 前面加上主事焦點用的前綴 ma-。上面的範例中也有這樣的例子，ina 例子中的 makawang 是走路，而 lima 例子中的 kemakawang 則是正在走路。

因此，或許未來若有機會繼續研究這項主題，可能要有一套規則來處理這種語法變化，如此一來應該能更精確地訓練出合適的模型。



### 9. 原住民語 NLP 研究之展望

原住民語這種 low resource 的語言在 NLP 領域並不好發揮，因此若能提升語料庫的大小，或許就能更好地推動相關發展。除了進行田野調查來蒐集資料，也可以在教學領域結合科技的力量，由族語老師指派作業，並且蒐集學員們回答的文章或錄音，經過評分或修正後加入資料庫中，如此一來既可以幫助教學現場的評量，也可以逐漸累積資料庫，以供未來的研究者使用。不過這只是我自己天馬行空的想法，不知道有沒有機會去實現。



### 10. 後記

我其實一直很想做原住民語相關的 NLP 研究，無奈目前資料庫真的太少了，適用於英文、中文這種強勢語言的研究方法，換到原住民語身上可能完全不能用。這次的期末專題給了我一個機會能試試這個領域的實驗，雖然結果仍然不盡理想，但也算是踏出第一步。

其實本次實驗中最困難的並不是實作 Word2Vec 的部分，從上面也看的出來，其實就是簡單幾行 code 而已。最困難的其實是 domain knowledge 的不足，我並不是甚麼族語專家，對於卑南語也僅僅是會基本對話的程度，要我看懂語料庫中的文章其實很困難，雖然有和語譯文，但譯文都是過度簡化過的版本，省略掉許多原文中的資訊。複雜的語法、不在族語詞典裡的詞等，使我要花更多的時間翻書，甚至詢問族語流利的族人來了解文章在寫甚麼。

儘管作的主題並不是多艱深的技術，也不是甚麼有即戰力的應用，但是花了大量的時間後終究產出了一點成果的感覺還是挺美妙的。希望未來原住民語相關的研究能更多，借助資訊的力量來幫助族語的傳承吧！



### References

https://arxiv.org/abs/1301.3781 (Word2Vec 原論文)

https://www.quora.com/What-are-the-main-differences-between-the-word-embeddings-of-ELMo-BERT-Word2vec-and-GloVe

https://radimrehurek.com/gensim/auto_examples/index.html#documentation

卑南語語法概論，鄧芳青
