import os, sys, pickle
import argparse, sys    # 명령 인자 받기

parser = argparse.ArgumentParser()
parser.add_argument('--train', help=' : train file', default=None)
parser.add_argument('--max_vocab', help=' : number of vocabularies', default=-1)
parser.add_argument('--vocab', help=' : rule file')
parser.add_argument('--infer', help=' : rule file', default=None)
parser.add_argument('--input', help=' : input txt file')
parser.add_argument('--output', help=' : output txt file (tokenized result)')
args = parser.parse_args()

MAX_VOCAB = int(args.max_vocab)         # base vocabulary 최대 개수
NEW_VOCAB_ITERATION = 20                # input 파일에 의한 새로운 vocabulary 생성 개수

vocabulary = []

# 심볼 '##'을 추가한다.
def add_subword_symbol(word):
    return '##' + word


'''
새로운 음절을 vocabulary 리스트에 추가한다.

이 때, 음절이 공백으로 구분된 것이 아닌 모음을 만나 음절로 분리된 것일 경우(syllable_flag가 True일 경우),

문자열 앞에 '##' 심볼을 추가하여 표기한다.

vocabulary 리스트에 음절을 추가한 뒤에 syllable_flag 값을 반환한다.
'''
def add_new_syllable(syllable, syllable_flag):
    if not syllable in vocabulary and syllable != '':
        if syllable_flag == True:                               # 자른 문자열이 공백으로 분리되지 않은 subword일 경우,
            syllable = add_subword_symbol(syllable)             # 문자열 앞에 '##' 심볼을 추가한다.

        vocabulary.append(syllable)                             # 음절을 vocabulary 리스트에 추가
        print(f"add vocab: ", syllable)

    return syllable_flag


'''
공백으로 나눈 단어들을 음절 단위로 분리해 vocabulary 리스트에 저장한다.

인자 element를 각 단어별로 loop문을 돌려 모음 철자가 등장할 때마다 syllable_index부터 i까지의 부분 문자열을 리스트에 저장한다.

이 때, 음절이 한 단어 안에서 공백이 아님에도 분리된 것이라면 앞에 '##' 심볼을 추가해 subword임을 표현한다.

루프를 빠져나온 후 추가되지 않은 마지막 문자열의 부분을 vocabulary 리스트에 추가하고 함수가 종료된다.
'''
def find_syllables(word):
    vowels = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']     #'모음'의 "모음"

    slice_index = 0         # 해당 인덱스부터 i까지의 subword를 저장
    syllable_flag = False   # element의 중간에 음절이 구분되었는지 표시하는 플래그
    syllable = ''

    for i in range(0, len(word)):
        if word[i] in vowels:
            syllable = word[slice_index:i + 1].replace('\n', '')
            syllable_flag = add_new_syllable(syllable, syllable_flag)

            syllable_flag = True
            slice_index = i + 1

    syllable = word[slice_index:len(word)].replace('\n', '')
    add_new_syllable(syllable, syllable_flag)


'''
vocabulary 리스트를 base vocabulary로 채운다.

train 파일을 공백을 기준으로 하여 줄마다 읽어들인 후 여러 개의 단어로 나누고,

그 단어들을 split_line 리스트에 저장한다. 이후 해당 리스트의 내용물(word)을

find_syllables()에 인자로 넘겨준다.

마지막으로 이를 pickle을 통해 바이너리 파일로 저장한다.
'''
def set_base_vocab():
    train_fp = open(args.train, "r")                                    # Train txt file pointer
    while True:
        if len(vocabulary) >= MAX_VOCAB: break;                         # MAX_VOCAB의 값보다 vocabulary 수가 많은 경우 생성을 멈춘다.
        line = train_fp.readline()
        if not line: break;                                             # 학습 텍스트를 모두 읽은 경우 생성을 멈춘다.
        split_line = line.split(' ')

        for word in split_line:                                         # 학습 텍스트 파일의 한 줄을 공백으로 나눈 단어를 하나씩
            find_syllables(word)                                        # find_syllables()에 인자로 넣어 실행한다.

    with open(args.vocab, "wb") as f:                                   # base vocabulary 생성이 끝나면 해당 리스트를 vocab 명령행 인자에서 받은 파일에
        pickle.dump(vocabulary, f)                                      # 저장한다.

    train_fp.close()

def slash_word_to_token(word):
    token_bench = []
    result_bench = []

    for token in vocabulary:
        if token in word:
            token_bench.append(token)

    for i in range(0, len(word) - 1):
        for j in range(i + 1, len(word)):
            if word[i:j] in token_bench:
                result_bench.append(word[i:j])

        result_bench.append('<unk>')

    return result_bench

def get_most_frequent_combination(word_bench):
    candidates_bench = {}
    for w_o_r_d in word_bench:
        for i in range(0, len(w_o_r_d) - 1):
            new_token = w_o_r_d[i] + w_o_r_d[i + 1]
            if new_token in candidates_bench:
                candidates_bench[new_token] = candidates_bench[new_token] + 1
            else:
                candidates_bench[new_token] = 1

    return max(candidates_bench, key=candidates_bench.get)

def add_new_token():
    while len(vocabulary) < MAX_VOCAB:
        shatterd_word_bench = {}
        train_fp = open(args.train, "r")

        while True:
            line = train_fp.readline()
            if not line: break;
            split_line = line.split(' ')

            for word in split_line:
                shatterd_word_list = slash_word_to_token(word)
                shatterd_word_bench.append(shatterd_word_list)

                if shatterd_word_list in shatterd_word_bench:
                    shatterd_word_bench[shatterd_word_list] = shatterd_word_bench[shatterd_word_list] + 1
                else:
                    shatterd_word_bench[shatterd_word_list] = 1

        winner = get_most_frequent_combination(shetterd_word_bench)
        print(f"newly added token: ", winner)
        vocabulary.append(winner)

        train_fp.close()


'''
input 텍스트 파일의 한 단어와 두 토큰을 인자로 전달받아 토큰의 조합이 단어 안에 포함되는지 확인한다.

만약 포함된다면, 토큰의 위치에 따라 적절한 처리를 한 뒤 이 새로운 토큰이 기존의 vocabulary 목록에 없었다면 토큰을 반환한다.

그 외의 모든 경우, <unk>를 반환한다.
'''
def find_vocab_in_word(word, vocab_1, vocab_2):
    new_vocab = vocab_1 + vocab_2                           # 토큰의 조합
    if new_vocab in word:
        if word.index(new_vocab) != 0:
            new_vocab = add_subword_symbol(new_vocab)

        if new_vocab not in vocabulary:                     # 해당 토큰이 기존 목록에 없을 경우, 이를 반환
            return new_vocab

    new_vocab = vocab_2 + vocab_1                           # 토큰의 조합(반대로)
    if new_vocab in word:
        if word.index(new_vocab) != 0:
            new_vocab = add_subword_symbol(new_vocab)

        if new_vocab not in vocabulary:                     # 해당 토큰이 기존 목록에 없을 경우, 이를 반환
            return new_vocab

    return '<unk>'


'''
vocabulary 리스트에서 두 개의 중복되지 않는 토큰을 뽑아 단어 안에 해당 조합이 나타나는지 확인한다.

토큰의 앞에 '##'심볼이 붙어있을 경우 이를 제거하고 find_vocab_in_word()에 인자로 넘겨 포함 여부를 확인한다.

그 결과를 반환하고, 만약 맞는 조합이 없다면 <unk>를 반환한다.
'''
def get_vocab_combinations(word):
    result = '<unk>'
    bench = []

    for element in vocabulary:
        if element in word:
            bench.append(element)


    for i in range(0, len(vocabulary) - 1):
        for j in range(i + 1, len(vocabulary)):
            vocab_1 = vocabulary[i]
            vocab_2 = vocabulary[j]
            if '##' in vocab_1:     vocab_1.replace('##', '')           # '##'심볼 제거
            if '##' in vocab_2:     vocab_2.replace('##', '')           # '##'심볼 제거

            result = find_vocab_in_word(word, vocab_1, vocab_2)

    return result

def split_word_base_vocab(word):
    return get_vocab_combinations(word)

'''
vocabulary 리스트의 토큰들을 output 텍스트 파일에 저장한다.
'''
def write_to_output():
    output_fp = open(args.output, "w")      # Output txt file pointer

    for element in vocabulary:
        output_fp.write(element)
        output_fp.write('\n')

    output_fp.close()


'''
BPE를 수행한다.

우선 base vocabulary 목록을 vocabulary 리스트에 저장한다.

input 텍스트 파일의 내용을 읽어 공백으로 나눈 단어를 vocabulary 리스트에 있는 subword의 두 개의 조합과 비교한다(find_subword() 이용).

subword의 조합이 단어 안에 포함될 경우 그 조합을 저장해두고, 텍스트 파일에서 해당 조합이 나온 횟수를 기록한다.

텍스트 파일 읽기가 끝나고, 기록된 조합 중 가장 많이 나타난 조합을 vocabulary 리스트에 저장하여 새로운 vocabulary를 만든다.

NEW_VOCAB_ITERATION만큼 새로운 vocabulary 생성 후, 이 토큰을 output 파일에 기록하고 변화된 토큰 목록을 infer 인자에 저장된 파일에 저장한다.
'''
def do_bpe():
    with open(args.infer, "rb") as f:           # infer 텍스트 파일을 불러온다.
        vocab = pickle.load(f)

    # infer 파일에서 가져온 base vocabulary 목록을 vocabulary 리스트에 저장한다.
    for element in vocab:
        vocabulary.append(element)

    for i in range(0, NEW_VOCAB_ITERATION):     # NEW_VOCAB_ITERATION만큼의 새로운 조합된 토큰을 생성한다.
        input_fp = open(args.input, "r")        # Input txt file pointer
        candidates_bench = {}                                                           # 새로 만들어진 토큰들을 저장할 dictionary
        while True:
            line = input_fp.readline()
            if not line: break;
            split_line = line.split(' ')

            for word in split_line:
                split_li = split_word_base_vocab(word)

                if new_vocab in candidates_bench:                                       # 새로운 토큰이 이미 candidates_bench에 존재할 경우,
                    candidates_bench[new_vocab] = candidates_bench[new_vocab] + 1       # 기존 토큰의 등장 횟수를 1 더한다.
                elif new_vocab == '<unk>':  continue                                    # <unk>일 경우 무시한다.
                else:
                    candidates_bench[new_vocab] = 1                                     # 새로운 토큰이 존재하지 않는다면, 추가한다.
            print(candidates_bench)

        final_candidate = max(candidates_bench, key=candidates_bench.get)               # 등장 횟수가 가장 많은 토큰을 뽑아서,
        print(f'final_candidates: ', final_candidate)
        vocabulary.append(final_candidate)                                              # 이를 vocabulary 리스트에 저장한다.

    write_to_output()                                                                   # Tokenization 수행 후, 새로운 토큰 리스트를
                                                                                        # output 텍스트 파일에 저장한다.

    with open(args.infer, "wb") as f:                                                   # 새로운 토큰이 추가된 vocabulary 리스트를
        pickle.dump(vocabulary, f)                                                      # infer 텍스트 파일에 저장한다.

    input_fp.close()


def main(argv, args):
    if args.train is not None:
        set_base_vocab()
        print("finish set_base_vocab()")
        print(f"len of vocab: ", len(vocabulary))
        add_new_token()
        print("finish add_new_token()")

    '''
    if args.infer is not None:
        print("start do_bpe()")
        do_bpe()
        print("finish do_bpe()")
    '''

if __name__ == '__main__':
    argv = sys.argv
    main(argv, args)
