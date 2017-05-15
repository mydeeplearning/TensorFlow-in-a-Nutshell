import tensorflow as tf
from tensorflow.contrib.layers import real_valued_column
from tensorflow.contrib.layers import bucketized_column
from tensorflow.contrib.layers import embedding_column
from tensorflow.contrib.layers import crossed_column
from tensorflow.contrib.layers import sparse_column_with_keys
from tensorflow.contrib.layers import sparse_column_with_hash_bucket
import numpy as np
import random
import pandas as pd
import os
from sklearn.model_selection import train_test_split


BATCH_SIZE = 32
ALPHA = 1e-4
CHECKPOINT_DIR = './tmp'


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


class Network(object):

    def __init__(self, wide_columns, deep_columns, holders_dict):
        # linear
        out_wide, _, _ = tf.contrib.layers.weighted_sum_from_feature_columns(
            columns_to_tensors=holders_dict,
            feature_columns=wide_columns,
            num_outputs=1
        )

        # dnn
        input_deep = tf.contrib.layers.input_from_feature_columns(
            columns_to_tensors=holders_dict,
            feature_columns=deep_columns
        )

        W_fc1 = weight_variable([input_deep.get_shape().as_list()[-1], 512])
        b_fc1 = bias_variable([512])
        h_fc1 = tf.nn.relu(tf.matmul(input_deep, W_fc1) + b_fc1)

        W_fc2 = weight_variable([512, 256])
        b_fc2 = bias_variable([256])
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

        W_fc3 = weight_variable([256, 128])
        b_fc3 = bias_variable([128])
        h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)

        W_fc4 = weight_variable([128, 1])
        b_fc4 = bias_variable([1])
        h_fc4 = tf.nn.relu(tf.matmul(h_fc3, W_fc4) + b_fc4)

        self.keep_prob = tf.placeholder(tf.float32)
        h_fc4_drop = tf.nn.dropout(h_fc4, self.keep_prob)

        h_merge = tf.add(out_wide, h_fc4_drop)
        self.out_p = tf.nn.sigmoid(h_merge)

        # loss function
        self.y = tf.placeholder(tf.float32, [None, 1], name='holder_y')
        self.loss = tf.losses.log_loss(self.y, self.out_p, epsilon=1e-15)
        return


class Trainer(object):

    def __init__(self):

        self.columns_categorical = ["Name", "Sex", "Embarked", "Cabin"]
        self.columns_continous = ["Age", "SibSp", "Parch", "Fare", "PassengerId", "Pclass"]
        self.columns = set(self.columns_categorical + self.columns_continous)

        self.wide_columns, self.deep_columns, self.holders_dict = self.create_feature_columns()
        self.net = Network(self.wide_columns, self.deep_columns, self.holders_dict)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=ALPHA)
        self.apply_gradient = self.optimizer.minimize(self.net.loss)

        self.session = tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True))
        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.tables_initializer())

        self.global_t = 0
        self.saver = tf.train.Saver()
        self.restore()
        return

    def create_feature_columns(self):
        """Build an estimator."""
        # Categorical columns
        sex = sparse_column_with_keys(column_name="Sex", keys=["female", "male"])
        embarked = sparse_column_with_keys(column_name="Embarked", keys=["C", "S", "Q"])

        cabin = sparse_column_with_hash_bucket("Cabin", hash_bucket_size=1000)
        name = sparse_column_with_hash_bucket("Name", hash_bucket_size=1000)

        # Continuous columns
        age = real_valued_column("Age")
        passenger_id = real_valued_column("PassengerId")
        sib_sp = real_valued_column("SibSp")
        parch = real_valued_column("Parch")
        fare = real_valued_column("Fare")
        p_class = real_valued_column("Pclass")

        # Transformations.
        age_buckets = bucketized_column(age, boundaries=[5, 18, 25, 30, 35, 40, 45, 50, 55, 65])
        # Wide columns and deep columns.
        wide_columns = [sex, embarked, cabin, name, age_buckets,
                        # crossed_column([age_buckets, sex], hash_bucket_size=int(1e6)),
                        # crossed_column([embarked, name], hash_bucket_size=int(1e4))
                        ]
        deep_columns = [
            embedding_column(sex, dimension=8),
            embedding_column(embarked, dimension=8),
            embedding_column(cabin, dimension=8),
            embedding_column(name, dimension=8),
            age,
            passenger_id,
            sib_sp,
            parch,
            fare,
            p_class
        ]

        feat_colums = wide_columns + deep_columns
        holders_dict = {}
        for c in feat_colums:
            if c.name not in self.columns:
                continue
            holder_name = 'holder_' + c.name
            holders_dict[c.name] = tf.placeholder(c.dtype, [None], name=holder_name)

        return wide_columns, deep_columns, holders_dict

    def create_feed_dict(self, df_batch, keep_prob=0.7):

        # continuous_cols = {k: tf.constant(df_batch[k].values) for k in self.columns_continous}

        # categorical_cols = {k: tf.SparseTensor(
        #     indices=[[i, 0] for i in range(df_batch[k].size)],
        #     values=df_batch[k].values,
        #     shape=[df_batch[k].size, 1])
        #     for k in self.columns_categorical}

        # feature = dict(continuous_cols)
        # feature.update(categorical_cols)

        feed_dict = {
            self.holders_dict['Name']: df_batch['Name'].values,
            self.holders_dict['Sex']: df_batch['Sex'].values,
            self.holders_dict['Embarked']: df_batch['Embarked'].values,
            self.holders_dict['Cabin']: df_batch['Cabin'].values,
            self.holders_dict['Age']: df_batch['Age'].values,
            self.holders_dict['SibSp']: df_batch['SibSp'].values,
            self.holders_dict['Parch']: df_batch['Parch'].values,
            self.holders_dict['Fare']: df_batch['Fare'].values,
            self.holders_dict['PassengerId']: df_batch['PassengerId'].values,
            self.holders_dict['Pclass']: df_batch['Pclass'].values,

            self.net.y: np.reshape(df_batch['Survived'].values, [-1, 1]),
            self.net.keep_prob: keep_prob
        }

        return feed_dict

    def run(self):

        df_data = pd.read_csv(tf.gfile.Open("./train.csv"), skipinitialspace=True)
        # df_test = pd.read_csv(tf.gfile.Open("./test.csv"), skipinitialspace=True)
        df_train, df_test = train_test_split(df_data, test_size=0.2)

        # train_size = df_train.shape[0]
        for epoch in range(int(1e7)):
            df_batch = df_train.sample(BATCH_SIZE)
            train_feed = self.create_feed_dict(df_batch, keep_prob=0.7)
            self.session.run(self.apply_gradient, feed_dict=train_feed)

            if epoch % 10001 == 0:
                self.backup()

            if epoch % 100 == 0:
                train_feed = self.create_feed_dict(df_train.sample(100), keep_prob=1.0)
                train_loss = self.session.run(self.net.loss, feed_dict=train_feed)

                test_feed = self.create_feed_dict(df_test.sample(100), keep_prob=1.0)
                test_loss = self.session.run(self.net.loss, feed_dict=test_feed)
                print 'epoch: %d, train_loss: %5f, test_loss: %5f' \
                    % (epoch + 1, train_loss, test_loss)

            # break
        return

    def restore(self):
        checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print("checkpoint loaded:", checkpoint.model_checkpoint_path)
            tokens = checkpoint.model_checkpoint_path.split("-")
            # set global step
            self.global_t = int(tokens[1])
            print(">>> global step set: ", self.global_t)
        else:
            print("Could not find old checkpoint")
        return

    def backup(self):
        if not os.path.exists(CHECKPOINT_DIR):
            os.mkdir(CHECKPOINT_DIR)

        self.saver.save(self.session, CHECKPOINT_DIR + '/' + 'checkpoint', global_step=self.global_t)
        return


def main():
    t = Trainer()
    t.run()

    # df_train = pd.read_csv(tf.gfile.Open("./train.csv"), skipinitialspace=True)
    # df_batch = df_train.sample(BATCH_SIZE)
    # # print df_batch.shape
    # # print df_batch['Survived']
    # print df_batch['Name'].values

    return


if __name__ == '__main__':
    main()
