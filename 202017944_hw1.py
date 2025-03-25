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
단어 하나를 받아 그 단어를 토큰 단위로 쪼갠다.

symbol_flag 인자는 활성화되었을 때 inference time에 쓰이는 것으로 간주되어 조건 만족 시 '##' 심볼을 추가한다.

최종적으로 쪼개진 토큰들을 튜플에 담아 반환한다.
'''
def slash_word_to_token(word, symbol_flag):
    result_bench = []                                               # 결과물을 저장할 리스트
    break_flag = False                                              # 루프 탈출용 플래그

    for i in range(0, len(word)):
        for j in range(i + 1, len(word) + 1):
            if word[i:j] in vocabulary:                             # 단어의 일부가 토큰과 일치할 때, 다음을 수행한다.

                if symbol_flag == True and i != 0:                  # 한 단어가 공백이 아닌 토큰으로 나누어지면, '##' 심볼을 추가한다.
                    result_bench.append('##' + word[i:j])           # 단, 이는 inference time에만 해당한다.
                else:
                    result_bench.append(word[i:j])

                i = j                                               # 추가한 토큰의 다음 부분부터 확인할 수 있도록 점프한다.
                break_flag = True
                break

        if break_flag == False:
            result_bench.append('<unk>')                            # 한 문자에 대해 대응하는 토큰을 찾지 못하면 <unk>를 결과에 추가

    return tuple(result_bench)


'''
새로운 음절을 vocabulary 리스트에 추가한다.
'''
def add_new_syllable(syllable):
    if not syllable in vocabulary and syllable != '':
        vocabulary.append(syllable)                             # 음절을 vocabulary 리스트에 추가
        print(f"add vocab: ", syllable)


'''
공백으로 나눈 단어들을 음절 단위로 분리해 vocabulary 리스트에 저장한다.

인자 word를 각 단어별로 loop문을 돌려 모음 철자가 등장할 때마다 syllable_index부터 i까지의 부분 문자열을 찾아,

해당 음절을 토큰으로 삼아 vocabulary에 저장한다.

루프를 빠져나온 후 추가되지 않은 마지막 문자열의 부분을 vocabulary 리스트에 추가하고 함수가 종료된다.
'''
def find_syllables(word):
    vowels = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']     #'모음'의 "모음"

    slice_index = 0         # 해당 인덱스부터 i까지의 subword를 저장
    syllable = ''

    for i in range(0, len(word)):
        if len(vocabulary) >= MAX_VOCAB:    return;
        if word[i] in vowels:
            syllable = word[slice_index:i + 1].replace('\n', '')                # 줄바꿈 문자가 있을 경우 제거한다.
            add_new_syllable(syllable)                                          # 음절을 토큰으로 추가
            slice_index = i + 1

    syllable = word[slice_index:len(word)].replace('\n', '')
    add_new_syllable(syllable)


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

    train_fp.close()

def get_most_frequent_combination(word_bench):
    candidates_bench = {}
    for w_o_r_d in word_bench:
        for i in range(0, len(w_o_r_d) - 1):
            new_token = w_o_r_d[i] + w_o_r_d[i + 1]
            if new_token in vocabulary:
                continue

            if new_token in candidates_bench:
                candidates_bench[new_token] = candidates_bench[new_token] + word_bench[w_o_r_d]
            else:
                candidates_bench[new_token] = word_bench[w_o_r_d]

    return max(candidates_bench, key=candidates_bench.get)


'''
base vocabulary를 완성한 후 토큰들을 합쳐 새로운 토큰을 생성한다.

MAX_VOCAB에 설정된 수만큼 vocabulary 리스트 개수가 만들어질 때까지 루프를 돌며,

단어마
'''
def add_new_token():
    while len(vocabulary) < MAX_VOCAB:
        shatterd_word_bench = {}
        train_fp = open(args.train, "r")

        while True:
            line = train_fp.readline()
            if not line: break;
            split_line = line.split(' ')

            for word in split_line:
                shatterd_word_tuple = slash_word_to_token(word, False)

                if shatterd_word_tuple in shatterd_word_bench:
                    shatterd_word_bench[shatterd_word_tuple] = shatterd_word_bench[shatterd_word_tuple] + 1
                else:
                    shatterd_word_bench[shatterd_word_tuple] = 1

        winner = get_most_frequent_combination(shatterd_word_bench)
        print(f"newly added token: ", winner)
        vocabulary.append(winner)

        train_fp.close()

'''
토큰화된 input 텍스트를 output 텍스트 파일에 저장한다.
'''
def write_to_output(fp, tokens):
    for token in tokens:를
        fp.write(token)
        fp.write(' ')


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

    input_fp = open(args.input, "r")        # Input txt file pointer
    output_fp = open(args.output, "w")
    while True:
        line = input_fp.readline()
        if not line: break;
        split_line = line.split(' ')

        for word in split_line:
            tokenized_word = slash_word_to_token(word, True)
            write_to_output(output_fp, tokenized_word)

        output_fp.write('\n')

    with open(args.infer, "wb") as f:                                                   # 새로운 토큰이 추가된 vocabulary 리스트를
        pickle.dump(vocabulary, f)                                                      # infer 텍스트 파일에 저장한다.

    input_fp.close()
    output_fp.close()


def main(argv, args):
    if args.train is not None:
        set_base_vocab()
        print("finish set_base_vocab()")
        print(f"len of vocab: ", len(vocabulary))
        add_new_token()
        print("finish add_new_token()")

        with open(args.vocab, "wb") as f:
            pickle.dump(vocabulary, f)

    if args.infer is not None:
        print("start do_bpe()")
        do_bpe()
        print("finish do_bpe()")

if __name__ == '__main__':
    argv = sys.argv
    main(argv, args)
