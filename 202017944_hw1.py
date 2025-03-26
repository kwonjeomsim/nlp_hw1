import os, sys, pickle, copy
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

vocabulary = []

# 심볼 '##'을 제거한다.
def remove_subword_symbol(word, symbol_flag):
    result = ''
    if symbol_flag == True:
        result = word.replace('##', '')
    else:
        result = word
    return result

'''
심볼 플래그가 True이면 토큰의 앞에 '##'심볼을 붙여 리턴한다.
'''
def determine_symbol(index, token, symbol_flag):
    if symbol_flag == True and index != 0:
        return '##' + token
    else:
        return token


'''
word 안에 token 문자열이 존재할 경우 호출되는 함수이다.

word에서 token 문자열과 일치하는 부분을 잘라 그 결과를 반환한다.
'''
def remove_token_in_word(word, token):
    index = 0
    while index < len(word):
        if word[index:index+len(token)] == token:
            break

        index = index + 1

    result_word = word[0:index] + word[index+len(token):len(word)]  # 원 문자열에서 token과 일치하는 부분을 제거한다.
    return result_word


'''
word 안에 존재하는 토큰들을 찾아 나타나는 순서 상관없이 반환한다.

길이로 내림차순 정렬된 vocabulary 리스트를 순차적으로 돌며 해당 토큰이 word 안에 존재할 경우 이를 결과 리스트에 추가한다.

이 때, symbol_flag에 따라 '##'심볼을 처리한다.

토큰을 추출한 후, 기존 단어에서 remove_token_in_word() 함수를 이용해 토큰이 나타나는 부분을 제거하여 다음 루프에 사용한다.

최종적으로 더 이상 단어 안에서 존재하는 토큰을 찾지 못하면 결과 리스트를 반환한다.
'''
def get_tokens_in_word(word, vocab_list, symbol_flag):
    result = []
    tmp_word = word
    for token in vocab_list:
        token_index = 0
        token = remove_subword_symbol(token, symbol_flag)       # symbol_flag가 True라면 '##'심볼을 제거한다.

        while token_index != -1:
            token_index = tmp_word.find(token, token_index)     # 토큰이 단어에서 나타나는 첫 인덱스를 저장한다.
            if token_index == -1:                               # 토큰이 단어 안에 없을 경우, -1을 반환한다.
                break

            result.append(determine_symbol(token_index, token, symbol_flag))

            new_tmp_word = remove_token_in_word(tmp_word, token)    # 토큰이 나타나는 부분을 원 단어에서 제거한다.
            tmp_word = new_tmp_word

    return result


'''
get_tokens_in_word()에서 생성한 토큰 후보군들을 정렬하고, <unk>를 적절히 추가해 토큰으로 쪼개진 단어의 리스트를 만든다.

단어의 맨 앞 글자부터 시작하여, 단어의 끝까지를 토큰과 비교하며 점점 부분 단어의 시작 인덱스를 1씩 늘린다.

이 방식으로 모든 토큰을 비교하며 부분 문자열 일치하는 토큰이 나오면 결과 리스트에 해당 토큰을 추가하고 후보군 리스트에서 제거한다.

만약 부분 문자열에 대해 모든 토큰과 비교해도 일치하는 것이 없을 경우에는 <unk> 토큰을 결과 리스트에 추가한다.
'''
def get_tokenized_word(word, result_bench, symbol_flag):
    result = []
    break_flag = False
    i = 0

    while i < len(word):
        tmp_word = word[i:len(word)]
        for token in result_bench:
            token_arg = remove_subword_symbol(token, symbol_flag)   #부분 문자열과 비교를 위해 '##'심볼이 있다면 제거한다.

            if tmp_word.find(token_arg) == 0:
                result.append(token)
                result_bench.remove(token)

                i = i + len(token_arg) - 1
                break_flag = True                   # 일치하는 토큰이 있으므로 break_flag를 True로 설정한다.
                break

        if break_flag == False:                     # 부분 문자열에 대해 일치하는 토큰이 없었음을 의미한다.
            result.append('<unk>')

        break_flag = False
        i = i + 1

    return result


'''
단어 하나를 받아 그 단어를 토큰 단위로 쪼갠다.

symbol_flag 인자는 활성화되었을 때 inference time에 쓰이는 것으로 간주되어 조건 만족 시 '##' 심볼을 추가한다.

최종적으로 쪼개진 토큰들을 튜플에 담아 반환한다.
'''
def slash_word_to_token(word, symbol_flag):
    result = []                                                 # 결과물을 저장할 리스트
    result_bench = []                                           # 임시 저장소

    sorted_vocabulary = sorted(vocabulary, key=lambda x: len(x), reverse=True)

    result_bench = get_tokens_in_word(word, sorted_vocabulary, symbol_flag)
    print(f"result_bench = ", result_bench)

    result = get_tokenized_word(word, result_bench, symbol_flag)
    print(f"result: ", result)
    return tuple(result)


'''
새로운 음절을 vocabulary 리스트에 추가한다.
'''
def add_new_syllable(syllable):
    if syllable not in vocabulary and syllable != '':
        vocabulary.append(syllable)                             # 음절을 vocabulary 리스트에 추가


'''
공백으로 나눈 단어들을 음절 단위로 분리해 vocabulary 리스트에 저장한다.

인자 word를 각 단어별로 for loop를 돌려 모음 철자가 등장할 때마다 마지막으로 끊은 지점부터 현재 지점까지의 문자열을 잘라서,

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
'''
def set_base_vocab():
    train_fp = open(args.train, "r")                                    # Train txt file pointer
    while True:
        if len(vocabulary) >= MAX_VOCAB: break

        line = train_fp.readline()
        if not line: break;                                             # 학습 텍스트를 모두 읽은 경우 생성을 멈춘다.
        split_line = line.split(' ')

        for word in split_line:                                         # 학습 텍스트 파일의 한 줄을 공백으로 나눈 단어를 하나씩
            find_syllables(word)                                        # find_syllables()에 인자로 넣어 실행한다.

    train_fp.close()


'''
새로 생긴 두 토큰의 조합 중 출현 빈도가 가장 높은 조합을 반환한다.
'''
def get_most_frequent_combination(word_bench):
    candidates_bench = {-1: -1}
    for w_o_r_d in word_bench:                          # 각각의 토큰을 뽑는다.
        for i in range(0, len(w_o_r_d) - 1):
            new_token = w_o_r_d[i] + w_o_r_d[i + 1]     # (0. 1), (1, 2), ..., (i, i+1)의 조합 생성
            if new_token in vocabulary:
                continue

            if new_token in candidates_bench:
                candidates_bench[new_token] = candidates_bench[new_token] + word_bench[w_o_r_d]
            else:
                candidates_bench[new_token] = word_bench[w_o_r_d]

    return max(candidates_bench, key=candidates_bench.get)  # 조합을 key로 하는 dictionary의 value값이 가장 높은 key 선택


'''
base vocabulary를 완성한 후 토큰들을 합쳐 새로운 토큰을 생성한다.

MAX_VOCAB에 설정된 수만큼 vocabulary 리스트 개수가 만들어질 때까지 루프를 돌며,

학습 데이터의 단어를 가지고 있는 토큰들로 쪼개서 가장 많이 나타난 토큰의 조합을 새로 토큰에 추가한다.
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
                shatterd_word_tuple = slash_word_to_token(word.replace('\n', ''), False)

                if shatterd_word_tuple in shatterd_word_bench:
                    shatterd_word_bench[shatterd_word_tuple] = shatterd_word_bench[shatterd_word_tuple] + 1
                else:
                    shatterd_word_bench[shatterd_word_tuple] = 1

        winner = get_most_frequent_combination(shatterd_word_bench)

        if winner == -1:    break
        vocabulary.append(winner)

        print(f"newly added token: ", winner)

        train_fp.close()

'''
토큰화된 input 텍스트를 output 텍스트 파일에 저장한다.
'''
def write_to_output(fp, tokens):
    for token in tokens:
        fp.write(token)
        fp.write(' ')           # 토큰들은 공백으로 구분한다.


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
    for token in vocab:
        vocabulary.append(token)

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

    input_fp.close()
    output_fp.close()


def main(argv, args):
    if args.train is not None:
        set_base_vocab()
        print("finish set_base_vocab()")
        add_new_token()
        print("finish add_new_token()")
        print(f"vocabulary: ", vocabulary)

        with open(args.vocab, "wb") as f:           # pickle을 이용해 완성된 학습 vocabulary를 저장
            pickle.dump(vocabulary, f)

    if args.infer is not None:
        do_bpe()
        print("finish do_bpe()")

if __name__ == '__main__':
    argv = sys.argv
    main(argv, args)
