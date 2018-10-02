import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score
import numpy as np


tf.set_random_seed(0)

class FM:
    def __init__(self,):
        self.X_dim =100
        self.Y_dim =1
        self.n_fields=10
        self.embed_size = 10
        self.n_epochs = 100
        self.learning_rate = 0.05
        self.reg=0
        self.reg_adv=1
        self.adv_type="grad"
        self.eps=0.5


    def create_placeholders(self):
        with tf.name_scope("data"):
            self.inputs = tf.placeholder(tf.int32, shape=[None,self.n_fields], name="inputs")
            self.labels= tf.placeholder(tf.float32, shape=[None,self.Y_dim], name="labels")

    def create_variables(self):
        with tf.name_scope("embedding"):
            self.embed_matrix=tf.Variable(tf.truncated_normal([self.X_dim,self.embed_size], mean=0.0, stddev=0.01),name="embed_matrix" )
            self.w=tf.Variable(tf.truncated_normal([self.X_dim,1], mean=0.0, stddev=0.01), name="w")
            self.b=tf.Variable(tf.truncated_normal( [1],mean=0.0, stddev=0.01), name="b")
            self.delta=tf.Variable(tf.zeros([self.X_dim,self.embed_size]),name="delta" )

    def create_inference(self):
        with tf.name_scope("inference"):
            self.xembed_matrix=tf.nn.embedding_lookup(self.embed_matrix,self.inputs)
            self.xw=tf.nn.embedding_lookup(self.w,self.inputs)
            self.p =0.5* tf.reduce_sum(
                # batch * k
                tf.square(tf.reduce_sum(self.xembed_matrix, 1)) -
                tf.reduce_sum(tf.square(self.xembed_matrix), 1),
                axis=1, keep_dims=True)
            #batch*1
            self.logits = tf.reduce_sum(self.xw, axis=1) + self.b + self.p
            self.preds = tf.sigmoid(self.logits)

    def create_inference_adv(self):
        with tf.name_scope("inference_adv"):
            self.xdelta_plus_embed=tf.nn.embedding_lookup(self.delta+self.embed_matrix,self.inputs)
            self.p_adv =0.5* tf.reduce_sum(
                # batch * k
                tf.square(tf.reduce_sum(self.xdelta_plus_embed, 1)) -
                tf.reduce_sum(tf.square(self.xdelta_plus_embed), 1),
                axis=1, keep_dims=True)
            #batch*1
            self.logits_adv = tf.reduce_sum(self.xw, axis=1) + self.b + self.p_adv
            self.preds_adv = tf.sigmoid(self.logits_adv)

    def create_loss(self):
        with tf.name_scope("loss"):
            self.log_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.logits))
            self.l2_loss = tf.reduce_sum(tf.square(self.embed_matrix))+tf.reduce_sum(tf.square(self.w))
            self.opt_loss = self.log_loss+self.l2_loss

    def create_loss_adv(self):
        with tf.name_scope("loss_adv"):
            self.log_loss_adv= tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.logits_adv))
            self.opt_loss_adv=self.opt_loss+self.reg_adv*self.log_loss_adv

    def create_adversarial(self):
        with tf.name_scope("adversarial"):
            # generate the adversarial weights by random method
            if self.adv_type == "random":
                # generation
                self.adv_embed = tf.truncated_normal(shape=[self.X_dim_, self.embed_size], mean=0.0, stddev=0.01)
                # normalization and multiply epsilon
                self.update_delta = self.delta.assign(tf.nn.l2_normalize(self.adv_embed, 1) * self.eps)
            # generate the adversarial weights by gradient-based method
            elif self.adv_type == "grad":
                self.grad_embed=tf.gradients(self.log_loss,self.embed_matrix)
                # convert the IndexedSlice Data to Dense Tensor !!!why
                self.grad_embed_dense = tf.stop_gradient(self.grad_embed)
                # normalization: new_grad = (grad / |grad|) * eps
                self.update_delta = self.delta.assign(tf.nn.l2_normalize(self.grad_embed_dense, 1) * self.eps)

    def create_optimizer(self):
        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate).minimize(self.opt_loss)
            self.optimizer_adv = tf.train.AdagradOptimizer(learning_rate=self.learning_rate).minimize(self.opt_loss_adv)

    def training(self,adv,dataset ,epoch_start,epoch_end, time_stamp, restore):  # saver is an object to save pq
        with tf.Session() as sess:
            # initialized the save op
            if adv:
                ckpt_save_path = "Pretrain/%s/AFM/%s/" % (dataset, time_stamp)
                ckpt_restore_path = "Pretrain/%s/FM/%s/" % (dataset,time_stamp)
            else:
                ckpt_save_path = "Pretrain/%s/FM/%s/" % (dataset, time_stamp)
                ckpt_restore_path = 0 if args.restore is None else "Pretrain/%s/MF_BPR/embed_%d/%s/" % (
                args.dataset, args.embed_size, args.restore)

            if not os.path.exists(ckpt_save_path):
                os.makedirs(ckpt_save_path)
            if ckpt_restore_path and not os.path.exists(ckpt_restore_path):
                os.makedirs(ckpt_restore_path)

            saver_ckpt = tf.train.Saver({'embedding_P': model.embedding_P, 'embedding_Q': model.embedding_Q})

            # pretrain or not
            sess.run(tf.global_variables_initializer())

            # restore the weights when pretrained
            if args.restore is not None or epoch_start:
                ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckpt_restore_path + 'checkpoint'))
                if ckpt and ckpt.model_checkpoint_path:
                    saver_ckpt.restore(sess, ckpt.model_checkpoint_path)