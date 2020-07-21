import tensorflow as tf
import numpy as np
from training_helper import linear_distribution, loss_function, coverage_loss
from utils import load_vocab_embedding_matrix
from configurations import Configurations
from model import Seq2seq


def training(model, train_dataset, epochs, learning_rate, pad_id):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    checkpoint = tf.train.Checkpoint(seq2seq=model)
    manager = tf.train.CheckpointManager(checkpoint, directory='./checkpoints', max_to_keep=3)

    teacher_forcing_ratio_distribution = linear_distribution(num_epoch=epochs)

    for epoch in range(epochs):
        teacher_forcing_ratio = teacher_forcing_ratio_distribution[epoch]
        epoch_loss = []
        for batch, (x, y) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                pred, _, att_list, cov_list = model.forward(x, y, teacher_forcing_ratio)
                loss = loss_function(y, pred, pad_id)
                if model.use_coverage:
                    loss += coverage_loss(att_list, cov_list)
            epoch_loss.append(loss.numpy())
            grads = tape.gradient(loss, model.variables)  # loss函数对模型的每一个参数求导
            optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))  # 更新参数
            if (batch + 1) % 50 == 0:
                print('[Epoch{} Batch{}] loss:{:.3f}'.format(epoch + 1, batch + 1, loss.numpy()))
        manager.save()  # 每个epoch后保存一个checkpoint
        print('Epoch{} Loss: {:.5f}'.format(epoch + 1, np.mean(epoch_loss)))
        print('***************')


if __name__ == '__main__':
    train_X = np.loadtxt('/data/train_X.txt', dtype='int')
    train_Y = np.loadtxt('/data/train_Y.txt', dtype='int')
    test_X = np.loadtxt('/data/test_X.txt', dtype='int')

    index2word, word2index, embedding_matrix = load_vocab_embedding_matrix()

    config = Configurations()

    train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_Y)).batch(config.batch_size)

    model = Seq2seq(vocab_size=embedding_matrix.shape[0],
                    embedding_dim=embedding_matrix.shape[1],
                    embedding_matrix=embedding_matrix,
                    gru_units=config.hid_dim,
                    dropout_rate=config.dropout)

    training(model, train_dataset, config.epochs, config.learning_rate, word2index['<PAD>'])
