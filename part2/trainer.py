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
import tempfile
from six.moves import urllib


BATCH_SIZE = 128
ALPHA = 0.00001
CHECKPOINT_DIR = './tmp'

LABEL_COLUMN = "label"
COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]


def maybe_download(train_data, test_data):
    """Maybe downloads training data and returns train and test file names."""
    if train_data:
        train_file_name = train_data
    else:
        train_file = tempfile.NamedTemporaryFile(delete=False)
        urllib.request.urlretrieve("http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.data",
                                   train_file.name)  # pylint: disable=line-too-long
        train_file_name = train_file.name
        train_file.close()
        print("Training data is downloaded to %s" % train_file_name)

    if test_data:
        test_file_name = test_data
    else:
        test_file = tempfile.NamedTemporaryFile(delete=False)
        urllib.request.urlretrieve("http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.test",
                                   test_file.name)  # pylint: disable=line-too-long
        test_file_name = test_file.name
        test_file.close()
        print("Test data is downloaded to %s" % test_file_name)

    return train_file_name, test_file_name


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


class Network(object):

    def __init__(self, wide_columns, deep_columns, holders_dict):
        self.linear_parent_scope = 'linear'
        self.dnn_parent_scope = 'dnn'
        # with tf.variable_scope(self.linear_parent_scope) as linear_input_scope:
        #     # linear
        #     out_wide, _, _ = tf.contrib.layers.weighted_sum_from_feature_columns(
        #         columns_to_tensors=holders_dict,
        #         feature_columns=wide_columns,
        #         num_outputs=1,
        #         weight_collections=[self.linear_parent_scope],
        #         scope=linear_input_scope
        #     )

        with tf.variable_scope(self.dnn_parent_scope) as dnn_input_scope:
            # dnn
            input_deep = tf.contrib.layers.input_from_feature_columns(
                columns_to_tensors=holders_dict,
                feature_columns=deep_columns,
                weight_collections=[self.dnn_parent_scope],
                scope=dnn_input_scope
            )

            W_fc1 = weight_variable([input_deep.get_shape().as_list()[-1], 100])
            b_fc1 = bias_variable([100])
            h_fc1 = tf.nn.relu(tf.matmul(input_deep, W_fc1) + b_fc1)

            W_fc2 = weight_variable([100, 50])
            b_fc2 = bias_variable([50])
            h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

            # W_fc3 = weight_variable([50, 128])
            # b_fc3 = bias_variable([128])
            # h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)

            self.keep_prob = tf.placeholder(tf.float32)
            h_fc2_drop = tf.nn.dropout(h_fc2, self.keep_prob)

            W_fc4 = weight_variable([50, 2])
            b_fc4 = bias_variable([2])
            self.readout = tf.matmul(h_fc2_drop, W_fc4) + b_fc4

            # h_merge = out_wide + h_fc4_drop
            # h_merge = h_fc4_drop
            # self.out_p = tf.nn.sigmoid(h_fc4_drop)

            # self.out_p = tf.nn.sigmoid(out_wide)

        # loss function
        self.y = tf.placeholder(tf.int32, [None], name='holder_y')
        self.y_ = tf.one_hot(self.y, 2, on_value=1, off_value=0, dtype=tf.int32)
        # self.loss = tf.losses.log_loss(self.y, self.out_p, epsilon=1e-15)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.readout))
        correct_prediction = tf.equal(tf.argmax(self.y_, 1), tf.argmax(self.readout, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return


class Trainer(object):

    def __init__(self):
        self.columns_categorical = ["workclass", "education", "marital_status", "occupation",
                                    "relationship", "race", "gender", "native_country"]
        self.columns_continous = ["age", "education_num", "capital_gain", "capital_loss",
                                  "hours_per_week"]
        # self.columns_categorical = ["Name", "Sex", "Embarked", "Cabin"]
        # self.columns_continous = ["Age", "SibSp", "Parch", "Fare", "PassengerId", "Pclass"]
        self.columns = set(self.columns_categorical + self.columns_continous)

        self.wide_columns, self.deep_columns, self.holders_dict = self.create_feature_columns()
        self.net = Network(self.wide_columns, self.deep_columns, self.holders_dict)

        # linear_optimizer = tf.train.FtrlOptimizer(learning_rate=0.2)
        # linear_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.net.linear_parent_scope)
        # # linear_vars = tf.trainable_variables()
        # linear_grad_and_vars = linear_optimizer.compute_gradients(self.net.loss, linear_vars)
        # linear_apply_grad = linear_optimizer.apply_gradients(linear_grad_and_vars)
        # self.apply_gradient = linear_apply_grad

        # dnn_optimizer = tf.train.AdamOptimizer(learning_rate=0.0005)
        # # dnn_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.net.dnn_parent_scope)
        # dnn_vars = tf.trainable_variables()
        # dnn_grad_and_vars = dnn_optimizer.compute_gradients(self.net.loss, dnn_vars)
        # dnn_apply_grad = dnn_optimizer.apply_gradients(dnn_grad_and_vars)
        # self.apply_gradient = dnn_apply_grad

        self.optimizer = tf.train.AdamOptimizer(learning_rate=ALPHA)
        self.apply_gradient = self.optimizer.minimize(self.net.loss)

        # self.apply_gradient = tf.group(*[linear_apply_grad, dnn_apply_grad])

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

        gender = tf.contrib.layers.sparse_column_with_keys(column_name="gender",
                                                           keys=["Female", "Male"])
        education = tf.contrib.layers.sparse_column_with_hash_bucket(
            "education", hash_bucket_size=1000)
        relationship = tf.contrib.layers.sparse_column_with_hash_bucket(
            "relationship", hash_bucket_size=100)
        workclass = tf.contrib.layers.sparse_column_with_hash_bucket(
            "workclass", hash_bucket_size=100)
        occupation = tf.contrib.layers.sparse_column_with_hash_bucket(
            "occupation", hash_bucket_size=1000)
        native_country = tf.contrib.layers.sparse_column_with_hash_bucket(
            "native_country", hash_bucket_size=1000)

        # Continuous base columns.
        age = tf.contrib.layers.real_valued_column("age")
        education_num = tf.contrib.layers.real_valued_column("education_num")
        capital_gain = tf.contrib.layers.real_valued_column("capital_gain")
        capital_loss = tf.contrib.layers.real_valued_column("capital_loss")
        hours_per_week = tf.contrib.layers.real_valued_column("hours_per_week")

        # Transformations.
        age_buckets = tf.contrib.layers.bucketized_column(age,
                                                          boundaries=[
                                                              18, 25, 30, 35, 40, 45,
                                                              50, 55, 60, 65
                                                          ])

        # Wide columns and deep columns.
        wide_columns = [
            gender, native_country, education, occupation, workclass,
            relationship, age_buckets,
            tf.contrib.layers.crossed_column([education, occupation],
                                             hash_bucket_size=int(1e4)),
            tf.contrib.layers.crossed_column(
                [age_buckets, education, occupation],
                hash_bucket_size=int(1e6)),
            tf.contrib.layers.crossed_column([native_country, occupation],
                                             hash_bucket_size=int(1e4))
        ]
        deep_columns = [
            tf.contrib.layers.embedding_column(workclass, dimension=8),
            tf.contrib.layers.embedding_column(education, dimension=8),
            tf.contrib.layers.embedding_column(gender, dimension=8),
            tf.contrib.layers.embedding_column(relationship, dimension=8),
            tf.contrib.layers.embedding_column(native_country,
                                               dimension=8),
            tf.contrib.layers.embedding_column(occupation, dimension=8),
            age,
            education_num,
            capital_gain,
            capital_loss,
            hours_per_week,
        ]

        feat_colums = wide_columns + deep_columns
        holders_dict = {}
        for c in feat_colums:
            if c.name not in self.columns:
                continue
            holder_name = 'holder_' + c.name
            holders_dict[c.name] = tf.placeholder(c.dtype, [None, 1], name=holder_name)
            # print holder_name
            # holders_dict[c.name] = tf.placeholder(tf.float32, [None], name=holder_name)
        print '---------------------------------------------------'
        print holders_dict
        print '---------------------------------------------------'
        return wide_columns, deep_columns, holders_dict

    def create_feed_dict(self, df_batch, keep_prob=0.7):

        feed_dict = {}
        for key in self.holders_dict:
            if key not in self.columns:
                continue
            feed_dict[self.holders_dict[key]] = np.reshape(df_batch[key].values, [-1, 1])

        feed_dict.update({
            # self.net.y: np.reshape(df_batch[LABEL_COLUMN].values, [-1, 1]),
            self.net.y: df_batch[LABEL_COLUMN].values.astype(np.int32),
            self.net.keep_prob: keep_prob
        })

        return feed_dict

    def run(self):

        df_train = pd.read_csv(tf.gfile.Open("./adult.data"), names=COLUMNS, skipinitialspace=True)
        df_test = pd.read_csv(tf.gfile.Open("./adult.test"), names=COLUMNS, skipinitialspace=True)

        df_train = df_train.dropna(how='any', axis=0)
        df_test = df_test.dropna(how='any', axis=0)

        df_train = df_train.sample(frac=1.0)
        df_test = df_test.sample(frac=1.0)

        df_train[LABEL_COLUMN] = (df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
        df_test[LABEL_COLUMN] = (df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)

        print '---------------------------------------------------'
        print 'pos:', np.sum(df_train[LABEL_COLUMN] == 1)
        print 'neg:', np.sum(df_train[LABEL_COLUMN] == 0)
        print '---------------------------------------------------'
        print 'pos:', np.sum(df_test[LABEL_COLUMN] == 1)
        print 'neg:', np.sum(df_test[LABEL_COLUMN] == 0)
        print '---------------------------------------------------'
        # return
        # train_size = df_train.shape[0]
        # df_test = pd.read_csv(tf.gfile.Open("./test.csv"), skipinitialspace=True)
        # df_train, df_test = train_test_split(df_data, test_size=0.2, random_state=0)

        # train_size = df_train.shape[0]
        print '===============checkpint: %d =========' % (self.global_t)
        while self.global_t < int(1e8):
            self.global_t += 1
            epoch = self.global_t
            df_batch = df_train.sample(BATCH_SIZE)
            train_feed = self.create_feed_dict(df_batch, keep_prob=1.0)
            # self.session.run(self.apply_gradient, feed_dict=train_feed)

            if epoch % 1000 == 0:
                self.backup()

            if epoch % 100 == 0:
                df_train_batch = df_train.sample(100)
                train_feed = self.create_feed_dict(df_train_batch, keep_prob=1.0)
                train_loss, train_accuracy = self.session.run([self.net.loss, self.net.accuracy], feed_dict=train_feed)
                # train_loss, train_pred = self.session.run([self.net.loss, self.net.out_p], feed_dict=train_feed)
                # train_accuracy = self.check_accuray(df_train_batch[LABEL_COLUMN].values, train_pred)

                df_test_batch = df_test.sample(100)
                test_feed = self.create_feed_dict(df_test_batch, keep_prob=1.0)
                test_loss, test_accuracy = self.session.run([self.net.loss, self.net.accuracy], feed_dict=test_feed)
                # test_loss, test_pred = self.session.run([self.net.loss, self.net.out_p], feed_dict=test_feed)
                # test_accuracy = self.check_accuray(df_train_batch[LABEL_COLUMN], test_pred)

                print 'epoch: %d, train_loss: %5f, train_accuracy: %5f, test_loss: %5f, test_accuracy: %5f' \
                    % (epoch, train_loss, train_accuracy, test_loss, test_accuracy)

            # if epoch == 100:
            #     break
        return

    def check_accuray(self, label, pred):
        pred = np.reshape(pred, [-1])
        size = np.shape(pred)[0]
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        correct = np.sum(pred == label)
        return float(correct) / size

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
    # maybe_download(None, None)
