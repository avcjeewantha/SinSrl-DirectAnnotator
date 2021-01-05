import os
import tensorflow as tf
import json

tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_resource_variables()

class BaseModel(object):
    """Generic class for general methods that are not specific to SRL"""

    def __init__(self, config):
        """Defines self.config and self.logger

        Args:
            config: (Config instance) class with hyper parameters,
                vocab and embeddings

        """
        self.config = config
        self.logger = config.logger
        self.sess = None
        self.saver = None

    def reinitialize_weights(self, scope_name):
        """Reinitializes the weights of a given layer"""
        variables = tf.compat.v1.get_variables(scope_name)
        init = tf.compat.v1.variables_initializer(variables)
        self.sess.run(init)

    def add_train_op(self, lr_method, lr, loss, clip=-1):
        """Defines self.train_op that performs an update on a batch
        Args:
            lr_method: (string) sgd method, for example "adam"
            lr: (tf.placeholder) tf.float32, learning rate
            loss: (tensor) tf.float32 loss to minimize
            clip: (python float) clipping of gradient. If < 0, no clipping

        """
        _lr_m = lr_method.lower()  # lower to make sure
        with tf.compat.v1.variable_scope("train_step"):
            if _lr_m == 'adam':  # sgd method
                optimizer = tf.compat.v1.train.AdamOptimizer(lr)
            elif _lr_m == 'adagrad':
                optimizer = tf.compat.v1.train.AdagradOptimizer(lr)
            elif _lr_m == 'sgd':
                optimizer = tf.compat.v1.train.GradientDescentOptimizer(lr)
            elif _lr_m == 'rmsprop':
                optimizer = tf.compat.v1.train.RMSPropOptimizer(lr)
            else:
                raise NotImplementedError("Unknown method {}".format(_lr_m))

            if clip > 0:  # gradient clipping if clip is positive
                grads, vs = zip(*optimizer.compute_gradients(loss))
                grads, gnorm = tf.clip_by_global_norm(grads, clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op = optimizer.minimize(loss)

    def initialize_session(self):
        """Defines self.sess and initialize the variables"""
        self.logger.info("Initializing tf session")
        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.saver = tf.compat.v1.train.Saver()

    def restore_session(self, dir_model):
        """Reload weights into session
        Args:
            sess: tf.Session()
            dir_model: dir with weights
        """
        self.logger.info("Reloading the latest trained model...")
        self.saver.restore(self.sess, dir_model)

    def save_session(self):
        """Saves session = weights"""
        if not os.path.exists(self.config.dir_model):
            os.makedirs(self.config.dir_model)
        self.saver.save(self.sess, self.config.dir_model)

        if not os.path.isfile(self.config.dir_model_root + 'modelResults.json') and not os.access(
                self.config.dir_model_root + 'modelsResults.json',
                os.R_OK):
            with open(self.config.dir_model_root + 'modelResults.json', 'w') as json_file:
                json.dump({"finalResults": [], "allParams": []}, json_file)  # write model stats into file
            json_file.close()

    def close_session(self):
        """Closes the session"""
        self.sess.close()
        tf.compat.v1.reset_default_graph()

    def add_summary(self):
        """Defines variables for Tensorboard
        Args:
            dir_output: (string) where the results are written
        """
        self.merged = tf.compat.v1.summary.merge_all()
        self.file_writer = tf.compat.v1.summary.FileWriter(self.config.dir_output,
                                                 self.sess.graph)

    def train(self, train, dev, test, best_score_got):
        """Performs training with early stopping and lr exponential decay
        Args:
            train: dataset that yields tuple of (sentences, tags)
            dev: dataset
            :param best_score_got: previous best score useful only when retrain first step. otherwise 0
        """
        param_dic = {}
        params = []
        best_score = best_score_got
        nepoch_no_imprv = 0  # for early stopping
        self.add_summary()  # tensorboard
        self.logger.info("no of layers {:}".format(self.config.layer))
        self.logger.info("no of steps {:}".format(self.config.step))
        self.logger.info("train_embeddings {:}".format(self.config.train_embeddings))
        self.logger.info("no of nepochs {:}".format(self.config.nepochs))
        self.logger.info("dropout {:}".format(self.config.dropout))
        self.logger.info("batch_size {:}".format(self.config.batch_size))
        self.logger.info("learning rate {:}".format(self.config.lr))
        self.logger.info("lr_decay {:}".format(self.config.lr_decay))
        self.logger.info("model_type {:}".format(self.config.model_type))
        param_dic["no of layers"] = self.config.layer
        param_dic["no of steps"] = self.config.step
        param_dic["train_embeddings"] = self.config.train_embeddings
        param_dic["no of nepochs"] = self.config.nepochs
        param_dic["dropout"] = self.config.dropout
        param_dic["batch_size"] = self.config.batch_size
        param_dic["learning rate"] = self.config.lr
        param_dic["lr_decay"] = self.config.lr_decay
        param_dic["model_type"] = self.config.model_type

        params.append(["no of layers {:}".format(self.config.layer),
                       "no of steps {:}".format(self.config.step),
                       "train_embeddings {:}".format(self.config.train_embeddings),
                       "no of nepochs {:}".format(self.config.nepochs),
                       "dropout {:}".format(self.config.dropout),
                       "batch_size {:}".format(self.config.batch_size),
                       "learning rate {:}".format(self.config.lr),
                       "lr_decay {:}".format(self.config.lr_decay),
                       "model_type {:}".format(self.config.model_type)])

        for epoch in range(self.config.nepochs):
            self.logger.info("Epoch {:} out of {:} of model no. - {:}".format(epoch + 1,
                                                                              self.config.nepochs, len(params)))
            self.logger.info(param_dic)

            score = self.run_epoch(train, dev, epoch, test)
            self.config.lr *= self.config.lr_decay  # decay learning rate

            # early stopping and saving best parameters
            if score >= best_score:
                nepoch_no_imprv = 0
                self.save_session()
                best_score = score
                param_dic["best_score"] = best_score
                self.logger.info("- new best score!")
                self.logger.info("Saved best model at epoch {}".format(epoch + 1))

                with open(self.config.dir_model + 'bestModelParams.txt', 'w') as outfile:
                    json.dump(param_dic, outfile)
                outfile.close()
            else:
                nepoch_no_imprv += 1
                if nepoch_no_imprv >= self.config.nepoch_no_imprv:
                    self.logger.info("- early stopping {} epochs without " \
                                     "improvement".format(nepoch_no_imprv))
                    break
        return params

    def evaluate(self, test):
        """Evaluate model on test set
        Args:
            test: instance of class Dataset
        """
        self.logger.info("Testing model over test set")
        metrics = self.run_evaluate(test)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                          for k, v in metrics.items()])
        self.logger.info(msg)
        return metrics
