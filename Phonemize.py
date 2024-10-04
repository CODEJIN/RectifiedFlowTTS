import logging, re, sys

from phonemizer import logger as phonemizer_logger
from phonemizer.backend import BACKENDS
from phonemizer.punctuation import Punctuation
from phonemizer.separator import default_separator, Separator
from phonemizer.phonemize import _phonemize

from typing import List, Dict, Union, Optional

from ko_pron import romanise
from pypinyin import pinyin
from pinyin_to_ipa import pinyin_to_ipa

from enum import Enum

logger = phonemizer_logger.get_logger()
logger.setLevel(logging.CRITICAL)

# English
english_phonemizer = BACKENDS['espeak'](
    language= 'en-us',
    punctuation_marks= Punctuation.default_marks(),
    preserve_punctuation= True,
    with_stress= False,
    tie= True,
    language_switch= 'remove-flags',
    words_mismatch= 'ignore',
    logger= logger
    )
whitespace_re = re.compile(r'\s+')
english_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
    ]]
def expand_abbreviations(text: str):
    for regex, replacement in english_abbreviations:
        text = re.sub(regex, replacement, text)

    return text

def _English_Phonemize(texts: Union[str, List[str]], chunk: int= 5000):
    if type(texts) == str:
        texts = [texts]

    texts = [text.lower() for text in texts]
    texts = [expand_abbreviations(text) for text in texts]

    pronunciations = []

    indices = range(0, len(texts), chunk)
    for index in indices:
        pronunciations_chunk = _phonemize(
            backend= english_phonemizer,
            text= texts[index:index + chunk],
            separator= Separator(phone= '|'), # default_separator,
            strip= True,
            njobs= 4,
            prepend_text= False,
            preserve_empty_lines= False
            )
        pronunciations.extend([
            re.sub(whitespace_re, ' ', pronunciation).replace('ː', '')
            for pronunciation in pronunciations_chunk
            ])

    return pronunciations

def English_Phoneme_Split(pronunciation: str) -> List[str]:
    # /ʔ/ is not vowel.
    # However, in English, the next vowel of /ʔ/ is usually silence /ə/.
    # To rule make, I manage /ʔ/ like a vowel
    diphones = ['tʃ', 'hw', 'aʊ', 'kw', 'eɪ', 'oʊ', 'dʒ', 'ɔɪ', 'aɪ']
    tie_bar = '\u0361'
    combining_vertical_line_below = '\u0329'

    phonemes = []
    index = 0
    while index < len(pronunciation):
        if index < len(pronunciation) - 6 and pronunciation[index + 1] == tie_bar and pronunciation[index + 3] == tie_bar and pronunciation[index + 5] == tie_bar:
            phonemes.append(pronunciation[index:index + 7].replace(tie_bar, ''))
            index += 7
        elif index < len(pronunciation) - 4 and pronunciation[index + 1] == tie_bar and pronunciation[index + 3] == tie_bar:
            phonemes.append(pronunciation[index:index + 5].replace(tie_bar, ''))
            index += 5
        elif index < len(pronunciation) - 2 and pronunciation[index + 1] == tie_bar:
            phonemes.append(pronunciation[index:index + 3].replace(tie_bar, ''))
            index += 3
        elif index < len(pronunciation) - 1 and pronunciation[index + 1] == combining_vertical_line_below:
            phonemes.append(pronunciation[index:index + 2])
            index += 2
        elif pronunciation[index:index + 2] in diphones:
            phonemes.append(pronunciation[index:index + 2])
            index += 2
        else:
            phonemes.append(pronunciation[index])
            index += 1

    if combining_vertical_line_below in phonemes:
        print(pronunciation)
        print(phonemes)
        assert False

    return phonemes

# Korean
def _Korean_Phonemize(texts: Union[str, List[str]]):
    if type(texts) == str:
        texts = [texts]

    pronunciations = [
        romanise(text, system_index= 'ipa')
        for text in texts
        ]
    
    pronunciations = [
        pronunciation if not '] ~ [' in pronunciation else pronunciation.split('] ~ [')[0]
        for pronunciation in pronunciations
        ]
    
    pronunciations = [
        pronunciation.replace('x', '')
        for pronunciation in pronunciations
        ]

    return pronunciations

korean_vowels = sorted(
    ['ja̠', 'wʌ̹', 'je̞', 'we̞', 'jʌ̹', 'wa̠', 'ɰi', 'ɥi', 'ʌ̹', 'a̠', 'o̞', 'e̞', 'jo', 'ø̞', 'ju', 'u', 'ɯ', 'i'],
    key= lambda x: len(x),
    reverse= True
    )
korean_consonants = sorted(
    ['b', 'd', 'd͡ʑ', 'h', 'k', 'kʰ', 'k͈', 'm', 'n', 'p', 'pʰ', 'p͈', 'sʰ', 's͈', 't', 'tʰ', 't̚', 't͈', 't͡ɕ', 't͡ɕʰ', 't͡ɕ͈', 'ŋ', 'ɡ', 'ɦ', 'ɭ', 'ɾ'],
    key= lambda x: len(x),
    reverse= True
    )

def _Korean_Split_Vowel(pronunciation: str):
    if len(pronunciation) == 0:
        return []

    for vowel in korean_vowels:
        if vowel in pronunciation:
            vowel_index = pronunciation.index(vowel)
            return \
                _Korean_Split_Vowel(pronunciation[:vowel_index]) + \
                [vowel] + \
                _Korean_Split_Vowel(pronunciation[vowel_index + len(vowel):])
    
    return [pronunciation]

def _Korean_Syllablize(pronunciation: str):
    pronunciation = _Korean_Split_Vowel(
        pronunciation= pronunciation
        ) # type: ignore
    
    syllables = []
    current_consonants = []
    for phoneme in pronunciation:
        if phoneme in korean_vowels:
            if len(syllables) > 0 and \
                (
                    len(current_consonants) > 1 or
                    (len(current_consonants) > 0 and current_consonants[0] in ['ɭ', 'ŋ'])
                    ):
                syllables[-1][2].append(current_consonants[0])
                current_consonants = current_consonants[1:]
            syllables.append((current_consonants, phoneme, []))
            current_consonants = []
        else:   # consonant
            if phoneme in korean_consonants:
                current_consonants.append(phoneme)
            else:
                is_applied = False
                for index in range(len(phoneme)):
                    if phoneme[:index] in korean_consonants and phoneme[index:] in korean_consonants:
                        current_consonants.append(phoneme[:index])
                        current_consonants.append(phoneme[index:])
                        is_applied = True
                        break
                assert is_applied, (pronunciation, phoneme)

    if len(current_consonants) > 0:
        syllables[-1][2].extend(current_consonants)

    return syllables


# Chinese
def _Chinese_Phonemize(
    texts: Union[str, List[str]],
    syllable_splitter: str= ' '
    ):
    if type(texts) == str:
        texts = [texts]

    
    return [
        f'{syllable_splitter}'.join([
            ''.join([
                x for x in pinyin_to_ipa(pinyin(character, v_to_u= True)[0][0])[0]
                ]).
                replace('˧', '').replace('˩', '').replace('˥', '').
                replace('tɕ', 't͡ɕ').replace('tɕʰ', 't͡ɕʰ').replace('ts', 't͡s').
                replace('tsʰ', 't͡sʰ').replace('tʂ', 't͡ʂ').replace('tʂʰ', 't͡ʂʰ').
                replace('ʈʂ', 'ʈ͡ʂ').replace('ʈʂʰ', 'ʈ͡ʂʰ')
            for character in text
            ])
        for text in texts        
        ]

chinese_diphthongs = ['ai̯','au̯','ou̯','ei̯']
chinese_monophthongs = ['ɛ','a','ʊ','i','ə','ɤ','o','ɔ','u','y','e']
chinese_syllablc_consonants = ['ɻ̩', 'ɹ̩', 'ɚ', 'n']  # 'for 嗯(n)'
chinese_onsets = sorted([
    't͡ɕʰ','t͡sʰ','t͡ʂʰ','ʈ͡ʂʰ','t͡ɕ','t͡s','t͡ʂ','ʈ͡ʂ',
    'kʰ','pʰ','tʰ','ʐ̩','ɕ','f','j','k','l','m','n',
    'p','ʐ','s','ʂ','t','w','x','ɥ','h','ʂ','ʐ','ɻ'
    ], key= lambda x: len(x), reverse= True)
chinese_codas = ['n','ŋ']

def _Chinese_Split_Phoneme(syllable: str):
    if len(syllable) == 0:
        return []

    for phoneme in chinese_diphthongs + chinese_monophthongs + chinese_syllablc_consonants + chinese_onsets + chinese_codas:
        if phoneme in syllable:
            phoneme_index = syllable.index(phoneme)
            return \
                _Chinese_Split_Phoneme(syllable[:phoneme_index]) + \
                [phoneme] + \
                _Chinese_Split_Phoneme(syllable[phoneme_index + len(phoneme):])

    return [syllable]

# Basically, Chinese does not consider onset, nucleus, coda. It uses intial + final.
# But, in multilingual condition, onset, nucleus, coda concept is usable for compatibility.
def _Chinese_Syllablize(pronunciation: str):
    pronunciation = [
        _Chinese_Split_Phoneme(syllable= x)
        for x in pronunciation.split(' ')
        ] # type: ignore

    syllables = []
    for syllable in pronunciation:
        is_exists_vowel = False
        for index, phoneme in enumerate(syllable):
            if phoneme in chinese_diphthongs + chinese_monophthongs + chinese_syllablc_consonants:
                syllables.append((syllable[:index], phoneme, syllable[index + 1:]))
                is_exists_vowel = True
                break
        assert is_exists_vowel, (pronunciation, syllable)    # no vowel.

    return syllables



# Japanese
def remove_initial_consonant(syllable):
    vowel_map = {
        'あ': 'あ', 'い': 'い', 'う': 'う', 'え': 'え', 'お': 'お',
        'か': 'あ', 'き': 'い', 'く': 'う', 'け': 'え', 'こ': 'お',
        'さ': 'あ', 'し': 'い', 'す': 'う', 'せ': 'え', 'そ': 'お',
        'た': 'あ', 'ち': 'い', 'つ': 'う', 'て': 'え', 'と': 'お',
        'な': 'あ', 'に': 'い', 'ぬ': 'う', 'ね': 'え', 'の': 'お',
        'は': 'あ', 'ひ': 'い', 'ふ': 'う', 'へ': 'え', 'ほ': 'お',
        'ま': 'あ', 'み': 'い', 'む': 'う', 'め': 'え', 'も': 'お',
        'や': 'あ', 'ゆ': 'う', 'よ': 'お',
        'ら': 'あ', 'り': 'い', 'る': 'う', 'れ': 'え', 'ろ': 'お',
        'わ': 'あ', 'を': 'お', 'ん': 'ん',
        'が': 'あ', 'ぎ': 'い', 'ぐ': 'う', 'げ': 'え', 'ご': 'お',
        'ざ': 'あ', 'じ': 'い', 'ず': 'う', 'ぜ': 'え', 'ぞ': 'お',
        'だ': 'あ', 'ぢ': 'い', 'づ': 'う', 'で': 'え', 'ど': 'お',
        'ば': 'あ', 'び': 'い', 'ぶ': 'う', 'べ': 'え', 'ぼ': 'お',
        'ぱ': 'あ', 'ぴ': 'い', 'ぷ': 'う', 'ぺ': 'え', 'ぽ': 'お'
    }
    
    return vowel_map.get(syllable, syllable)

def _Japanese_Syllable_to_IPA(
    syllable: str,
    next_syllable: Optional[str]= None  # for 'ん', 'っ'
    ):
    ipa_dict = {
        'あ'  : [[   ], 'a', []], 'い': [[    ], 'i', []],  'う'  : [[    ], 'ɯ', []],  'え': [[   ], 'e', []], 'お'  : [[    ], 'o', []],
        'か'  : [['k'], 'a', []], 'き': [['k' ], 'i', []],  'く'  : [['k' ], 'ɯ', []],  'け': [['k'], 'e', []], 'こ'  : [['k',], 'o', []],
        'さ'  : [['s'], 'a', []], 'し': [['ɕ' ], 'i', []],  'す'  : [['s' ], 'ɯ', []],  'せ': [['s'], 'e', []], 'そ'  : [['s',], 'o', []],
        'た'  : [['t'], 'a', []], 'ち': [['tɕ'], 'i', []],  'つ'  : [['ts'], 'ɯ', []],  'て': [['t'], 'e', []], 'と'  : [['t',], 'o', []],
        'な'  : [['n'], 'a', []], 'に': [['ɲ' ], 'i', []],  'ぬ'  : [['n' ], 'ɯ', []],  'ね': [['n'], 'e', []], 'の'  : [['n',], 'o', []],
        'は'  : [['h'], 'a', []], 'ひ': [['ç' ], 'i', []],  'ふ'  : [['ɸ' ], 'ɯ', []],  'へ': [['h'], 'e', []], 'ほ'  : [['h',], 'o', []],
        'ま'  : [['m'], 'a', []], 'み': [['m' ], 'i', []],  'む'  : [['m' ], 'ɯ', []],  'め': [['m'], 'e', []], 'も'  : [['m',], 'o', []],
        'や'  : [['j'], 'a', []],                           'ゆ'  : [['j' ], 'ɯ', []],                          'よ'  : [['j',], 'o', []],
        'ら'  : [['ɾ'], 'a', []], 'り': [['ɾ' ], 'i', []],  'る'  : [['ɾ' ], 'ɯ', []],  'れ': [['ɾ'], 'e', []], 'ろ'  : [['ɾ',], 'o', []],
        'わ'  : [['ɰ'], 'a', []],                                                                               'を'  : [[    ], 'o', []],

                                                            'ゔ'  : [['v',], 'u', []], 
        'が'  : [['g' ], 'a', []], 'ぎ': [['g' ], 'i', []], 'ぐ'  : [['g',], 'ɯ', []], 'げ': [['g' ], 'e', []], 'ご'  : [['g',], 'o', []],
        'ざ'  : [['dz'], 'a', []], 'じ': [['dʑ'], 'i', []], 'ず'  : [['dz'], 'ɯ', []], 'ぜ': [['dz'], 'e', []], 'ぞ'  : [['dz'], 'o', []],
        'だ'  : [['d' ], 'a', []], 'ぢ': [['dʑ'], 'i', []], 'づ'  : [['dz'], 'ɯ', []], 'で': [['d' ], 'e', []], 'ど'  : [['d',], 'o', []],
        'ば'  : [['b' ], 'a', []], 'び': [['b' ], 'i', []], 'ぶ'  : [['b',], 'ɯ', []], 'べ': [['b' ], 'e', []], 'ぼ'  : [['b',], 'o', []],

        'ぱ'  : [['p' ], 'a', []], 'ぴ': [['p' ], 'i', []], 'ぷ'  : [['p',], 'ɯ', []], 'ぺ': [['p' ], 'e', []], 'ぽ'  : [['p',], 'o', []],

        'きゃ': [['kʲ'], 'a', []],                          'きゅ': [['kʲ'], 'ɯ', []],                          'きょ': [['kʲ'], 'o', []],
        'しゃ': [['ɕ' ], 'a', []],                          'しゅ': [['ɕ' ], 'ɯ', []],                          'しょ': [['ɕ' ], 'o', []],
        'ちゃ': [['tɕ'], 'a', []],                          'ちゅ': [['tɕ'], 'ɯ', []],                          'ちょ': [['tɕ'], 'o', []],
        'にゃ': [['ɲ' ], 'a', []],                          'にゅ': [['ɲ' ], 'ɯ', []],                          'にょ': [['ɲ' ], 'o', []],
        'ひゃ': [['ç' ], 'a', []],                          'ひゅ': [['ç' ], 'ɯ', []],                          'ひょ': [['ç' ], 'o', []],
        'みゃ': [['mʲ'], 'a', []],                          'みゅ': [['mʲ'], 'ɯ', []],                          'みょ': [['mʲ'], 'o', []],
        'りゃ': [['ɾʲ'], 'a', []],                          'りゅ': [['ɾʲ'], 'ɯ', []],                          'りょ': [['ɾʲ'], 'o', []],
        'ぎゃ': [['gʲ'], 'a', []],                          'ぎゅ': [['gʲ'], 'ɯ', []],                          'ぎょ': [['gʲ'], 'o', []],
        'じゃ': [['dʑ'], 'a', []],                          'じゅ': [['dʑ'], 'ɯ', []],                          'じょ': [['dʑ'], 'o', []],
        'びゃ': [['bʲ'], 'a', []],                          'びゅ': [['bʲ'], 'ɯ', []],                          'びょ': [['bʲ'], 'o', []],
        'ぴゃ': [['pʲ'], 'a', []],                          'ぴゅ': [['pʲ'], 'ɯ', []],                          'ぴょ': [['pʲ'], 'o', []],

        'ぁ'  : [[   ], 'a', []], 'ぃ': [[    ], 'i', []],  'ぅ'  : [[    ], 'ɯ', []],  'ぇ': [[   ], 'e', []], 'ぉ'  : [[    ], 'o', []],
        }
    
    if syllable in ipa_dict.keys():
        return ipa_dict[syllable]
    elif len(syllable) > 1 and syllable[-1] == 'っ':
        ipa = _Japanese_Syllable_to_IPA(syllable[:-1], 'っ')
        next_onset = ipa_dict[next_syllable[0]][0] if not next_syllable is None else []
        if len(next_onset) == 0:
            next_onset.append('ʔ')
        ipa[2] = next_onset
        return ipa
    elif len(syllable) > 1 and syllable[-1] == 'ん':
        ipa = _Japanese_Syllable_to_IPA(syllable[:-1], 'ん')        
        _, _, ipa[2] = _Japanese_Syllable_to_IPA(syllable[-1], next_syllable)
        return ipa
    elif len(syllable) == 2:
        ipa = ipa_dict[syllable[0]]
        if len(ipa[2]) != 0:
            raise NotImplementedError(f'unimplemented syllable: {syllable}')
        elif syllable[1] in ['ぁ', 'ぃ', 'ぅ', 'ぇ', 'ぉ']:
            stegana_ipa = ipa_dict[syllable[1]]
            ipa[0] = ipa[0] + stegana_ipa[0]
            ipa[1] = stegana_ipa[1]        
        else: # syllable[1] in ['ゕ', 'ゖ', 'ゎ', 'ゃ', 'ゅ', 'ょ']
            raise NotImplementedError(f'unimplemented syllable: {syllable}')        
        return ipa    
    elif len(syllable) == 3 and syllable[2] == 'ん':
        ipa = _Japanese_Syllable_to_IPA(syllable[:2], syllable[2])
        _, _, ipa[2] = _Japanese_Syllable_to_IPA(syllable[2], next_syllable)
        return ipa
    
    if syllable == 'ん':    # single 'ん'
        if next_syllable is None:
            return [[], '', ['ɴ']]
        elif next_syllable[0] in [
            'あ', 'い', 'う', 'え', 'お',
            'な', 'に', 'ぬ', 'ね', 'の',
            'ら', 'り', 'る', 'れ', 'ろ',
            'ざ', 'じ', 'ず', 'ぜ', 'ぞ',
            'だ', 'ぢ', 'づ', 'で', 'ど'
            ]:
            return [[], '', ['n']]
        elif next_syllable[0] in [
            'は', 'ひ', 'ふ', 'へ', 'ほ',
            'ま', 'み', 'む', 'め', 'も',
            'ば', 'び', 'ぶ', 'べ', 'ぼ',
            'ぱ', 'ぴ', 'ぷ', 'ぺ', 'ぽ'
            ]:
            return [[], '', ['m']]
        elif next_syllable[0] in [
            'か', 'き', 'く', 'け', 'こ',
            'が', 'ぎ', 'ぐ', 'げ', 'ご'
            ]:
            return [[], '', ['ŋ']]
        elif next_syllable[0] in [
            'や', 'ゆ', 'よ', 'わ', 'を'
            ]:
            return [[], '', ['ɰ̃']]
        elif next_syllable[0] in [
            'さ', 'し', 'す', 'せ', 'そ',
            'た', 'ち', 'つ', 'て', 'と'
            ]:
            return [[], '', ['n̪']]
        else:
            raise ValueError(f'Unsupported next syllable: {next_syllable}')

    raise NotImplementedError(f'unimplemented syllable: {syllable}, {next_syllable}')

def _Japanese_Phonemize(texts: Union[str, List[str]]):
    if type(texts) == str:
        texts = [texts]

    pronunciations = []

    indices = range(0, len(texts), chunk)
    for index in indices:
        
        pronunciations.extend(pronunciations_chunk)

    return pronunciations

def _Japanese_Split_Phoneme(syllable: str):
    if len(syllable) == 0:
        return []

    for phoneme in chinese_diphthongs + chinese_monophthongs + chinese_syllablc_consonants + chinese_onsets + chinese_codas:
        if phoneme in syllable:
            phoneme_index = syllable.index(phoneme)
            return \
                _Chinese_Split_Phoneme(syllable[:phoneme_index]) + \
                [phoneme] + \
                _Chinese_Split_Phoneme(syllable[phoneme_index + len(phoneme):])

    return [syllable]

def _Japanese_Syllablize(pronunciation: str):
    pronunciation = [
        _Japanese_Split_Phoneme(syllable= x)
        for x in pronunciation.split(' ')
        ] # type: ignore

    syllables = []
    for syllable in pronunciation:
        is_exists_vowel = False
        for index, phoneme in enumerate(syllable):
            if phoneme in chinese_diphthongs + chinese_monophthongs + chinese_syllablc_consonants:
                syllables.append((syllable[:index], phoneme, syllable[index + 1:]))
                is_exists_vowel = True
                break
        assert is_exists_vowel, (pronunciation, syllable)    # no vowel.

    return syllables





class Language(Enum):
    Korean= 0
    English= 1
    Chinese= 2
    Japanese= 3

def Phonemize(texts: Union[str, List[str]], language: Language):
    if language == Language.Korean:
        return _Korean_Phonemize(texts)
    elif language == Language.English:
        return _English_Phonemize(texts)
    elif language == Language.Chinese:
        return _Chinese_Phonemize(texts)
    elif language == Language.Japanese:
        return _Japanese_Phonemize(texts)
    else:
        raise NotImplementedError(f'Unsupported language: {language}')