import tensorflow as tf
import random


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix, enc_units, dropout_rate):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size,
                                                   output_dim=embedding_dim,
                                                   weights=[embedding_matrix],
                                                   trainable=False)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.gru = tf.keras.layers.GRU(enc_units, return_sequences=True, return_state=True)

    def call(self, inputs, training):
        x = self.embedding(inputs)
        x = self.dropout(x, training=training)  # 一定要写training=True才会起作用
        outputs, hidden = self.gru(x)
        return outputs, hidden


class Attention(tf.keras.Model):
    def __init__(self, units, use_coverage=False):
        super().__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
        if use_coverage:
            self.W3 = tf.keras.layers.Dense(units)

    def call(self, query, values, prev_coverage=None):
        # query  [batch_size, gru_units]
        # values [batch_size, seq_len, gru_units]
        query = tf.expand_dims(query, 1)
        if prev_coverage == None:
            score = self.V(tf.nn.tanh(self.W1(values) + self.W2(query)))  # [batch_size, seq_len, 1]
        else:
            prev_coverage = tf.expand_dims(prev_coverage, 1)
            score = self.V(
                tf.nn.tanh(self.W1(values) + self.W2(query) + self.W3(prev_coverage)))  # [batch_size, seq_len, 1]
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values  # [batch_size, seq_len, gru_units]
        context_vector = tf.reduce_sum(context_vector, axis=1)  # [batch_szie, gru_units]

        return attention_weights, context_vector  # [batch_szie, gru_units]


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix, dec_units, dropout_rate, use_coverage=False):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size,
                                                   output_dim=embedding_dim,
                                                   weights=[embedding_matrix],
                                                   trainable=False)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.attention = Attention(dec_units, use_coverage)
        self.gru = tf.keras.layers.GRU(dec_units, return_state=True, return_sequences=True)

        self.embedding2vocab1 = tf.keras.layers.Dense(dec_units * 2)
        self.embedding2vocab2 = tf.keras.layers.Dense(dec_units * 4)
        self.embedding2vocab3 = tf.keras.layers.Dense(vocab_size, activation='softmax')

    def call(self, inputs, enc_outputs, dec_hidden, training, prev_coverage=None):
        attention_weights, context_vector = self.attention(dec_hidden, enc_outputs,
                                                           prev_coverage)  # context_vector(batch_size, enc_units)
        x = self.embedding(inputs)  # embeded x (batch_size,1,embedding_dim)
        x = self.dropout(x, training=training)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)  # x(batch_size, 1, embedding_dim+enc_units)
        dec_outputs, dec_hidden = self.gru(x)

        output = self.embedding2vocab1(dec_hidden)
        output = self.embedding2vocab2(output)
        prediction = self.embedding2vocab3(output)

        return prediction, dec_hidden, attention_weights


class Seq2seq(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix, gru_units, dropout_rate, use_coverage=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.use_coverage = use_coverage
        self.encoder = Encoder(vocab_size, embedding_dim, embedding_matrix, gru_units, dropout_rate)
        self.decoder = Decoder(vocab_size, embedding_dim, embedding_matrix, gru_units, dropout_rate, self.use_coverage)

    def forward(self, inputs, target, teacher_forcing_ratio):
        batch_size = target.shape[0]
        seq_len = target.shape[1]

        enc_outputs, enc_hidden = self.encoder(inputs, training=True)
        # 初始化存结果的张量
        outputs = tf.zeros([batch_size, 1, self.vocab_size])
        pred_ids = tf.zeros([batch_size, 1], dtype=tf.int64)
        # t1时刻的decoder输入
        dec_hidden = enc_hidden
        dec_ipt = tf.expand_dims(target[:, 0], 1)  # <BOS>

        prev_coverage = tf.zeros([batch_size, inputs.shape[1]])
        att_list = []
        cov_list = []

        for t in range(1, seq_len):
            if self.use_coverage:
                pred, dec_hidden, att_weights = self.decoder(dec_ipt, enc_outputs, dec_hidden, training=True,
                                                             prev_coverage=prev_coverage)
                prev_coverage += tf.squeeze(att_weights)
                att_list.append(tf.squeeze(att_weights))
                cov_list.append(prev_coverage)
            else:
                pred, dec_hidden, _ = self.decoder(dec_ipt, enc_outputs, dec_hidden, training=True)

            pred_id = tf.argmax(pred, axis=1)

            outputs = tf.concat([outputs, tf.expand_dims(pred, 1)], axis=1)
            pred_ids = tf.concat([pred_ids, tf.expand_dims(pred_id, 1)], axis=1)

            # random()在[0.0, 1.0)范围内生成随机数。
            # teacher_forcing_ratio [0.0, 1.0]随着epoch增加而减小
            # teacher forcing的概率越来越小
            teacher_force = random.random() <= teacher_forcing_ratio
            dec_ipt = tf.expand_dims(target[:, t], 1) if teacher_force else tf.expand_dims(pred_ids[:, t], 1)

        return outputs, pred_ids, att_list, cov_list

    def inference(self, inputs, bos_index, eos_index, max_len):
        batch_size = inputs.shape[0]

        enc_outputs, enc_hidden = self.encoder(inputs, training=False)
        # 初始化存结果的张量
        outputs = tf.zeros([batch_size, 1, self.vocab_size])
        pred_ids = tf.zeros([batch_size, 1], dtype=tf.int64)
        # t1时刻的decoder输入
        dec_hidden = enc_hidden
        dec_ipt = tf.expand_dims([bos_index] * batch_size, 1)  # <BOS>

        prev_coverage = tf.zeros([batch_size, inputs.shape[1]])

        for t in range(1, max_len):
            if self.use_coverage:
                pred, dec_hidden, att_weights = self.decoder(dec_ipt, enc_outputs, dec_hidden, training=True,
                                                             prev_coverage=prev_coverage)
                prev_coverage += tf.squeeze(att_weights)
            else:
                pred, dec_hidden, _ = self.decoder(dec_ipt, enc_outputs, dec_hidden, training=False)

            pred_id = tf.argmax(pred, axis=1)

            pred_ids = tf.concat([pred_ids, tf.expand_dims(pred_id, 1)], axis=1)

            dec_ipt = tf.expand_dims(pred_ids[:, t], 1)

        return pred_ids

    def beam_inference(self, inputs, bos_index, eos_index, max_len, beam_size):
        enc_ipt = tf.reshape(inputs, [1, -1])

        enc_outputs, enc_hidden = self.encoder(enc_ipt, training=False)
        dec_hidden = enc_hidden

        outputs = [
            [[bos_index], 1.0]
        ]

        for t in range(1, max_len):
            candidates = []
            for i in outputs:
                seq, prob = i

                # if seq[-1] == eos_index:
                #   return [seq]

                dec_ipt = tf.reshape(seq[-1], [1, -1])
                pred, dec_hidden, _ = self.decoder(dec_ipt, enc_outputs, dec_hidden, training=False)
                top_k_probs, top_k_ids = tf.nn.top_k(tf.squeeze(pred), k=beam_size)

                for j in range(beam_size):
                    candidate = [
                        seq + [top_k_ids[j].numpy()], prob * top_k_probs[j].numpy()
                    ]
                    candidates.append(candidate)

            sorted_candidates = sorted(candidates, key=lambda p: p[1], reverse=True)  # sorted从小到大排序
            outputs = sorted_candidates[:beam_size]

        return [candidate[0] for candidate in outputs]
