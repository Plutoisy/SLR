import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense

# 数据准备（这里只是一个示例，实际需要使用真实的中英文平行语料库）
english_texts = ['am', 'I', 'happy']
chinese_texts = ['是', '我', '高兴']

# 在每个目标文本前后添加起始和结束标志
chinese_texts = ['\t' + text + '\n' for text in chinese_texts]

# 创建词汇表并进行向量化
english_vocab = set(' '.join(english_texts))
chinese_vocab = set(''.join(chinese_texts))

english_token_index = {char: i for i, char in enumerate(english_vocab)}
chinese_token_index = {char: i for i, char in enumerate(chinese_vocab)}

max_encoder_seq_length = max([len(txt) for txt in english_texts])
max_decoder_seq_length = max([len(txt) for txt in chinese_texts])

num_encoder_tokens = len(english_vocab)
num_decoder_tokens = len(chinese_vocab)

encoder_input_data = np.zeros(
    (len(english_texts), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
decoder_input_data = np.zeros(
    (len(chinese_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
decoder_target_data = np.zeros(
    (len(chinese_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

for i, (input_text, target_text) in enumerate(zip(english_texts, chinese_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, english_token_index[char]] = 1.
    for t, char in enumerate(target_text):
        decoder_input_data[i, t, chinese_token_index[char]] = 1.
        if t > 0:
            decoder_target_data[i, t - 1, chinese_token_index[char]] = 1.

# 构建模型
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder_lstm = LSTM(256, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# 编译模型
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# 训练模型
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=64,
          epochs=500,
          validation_split=0.2)

# 构建推理模型
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(256,))
decoder_state_input_c = Input(shape=(256,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# 翻译函数
def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, chinese_token_index['\t']] = 1.

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = list(chinese_token_index.keys())[list(chinese_token_index.values()).index(sampled_token_index)]
        decoded_sentence += sampled_char

        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        states_value = [h, c]

    return decoded_sentence

# 测试翻译
english_texts = ["I","am","happy"]
for seq_index in range(len(english_texts)):
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('Input sentence:', english_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)
