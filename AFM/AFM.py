import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score
import numpy as np
from datasets import iPinYou
from time import time
from time import strftime
from time import localtime
import os
tf.set_random_seed(0)

class FM:
    def __init__(self,):
        self.X_dim =937670
        self.Y_dim =1
        self.n_fields=16
        self.embed_size = 20
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
            self.delta=tf.Variable(tf.zeros([self.X_dim,self.embed_size]),name="delta" ,trainable=False)

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
                self.grad_embed=tf.gradients(self.log_loss,self.embed_matrix)[0]#!!!!!!!!
                # convert the IndexedSlice Data to Dense Tensor !!!why
                self.grad_embed_dense = tf.stop_gradient(self.grad_embed)
                # normalization: new_grad = (grad / |grad|) * eps
                self.update_delta = self.delta.assign(tf.nn.l2_normalize(self.grad_embed_dense, 1) * self.eps)

    def create_optimizer(self):
        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate).minimize(self.opt_loss)
            self.optimizer_adv = tf.train.AdagradOptimizer(learning_rate=self.learning_rate).minimize(self.opt_loss_adv)

    def build_graph(self):
        self.create_placeholders()
        self.create_variables()
        self.create_inference()
        self.create_inference_adv()
        self.create_loss()
        self.create_adversarial()
        self.create_loss_adv()
        self.create_optimizer()


def training(model,dataset ,epoch_start,epoch_end,time_stamp=None, adv=False,restore_time=None ):  # saver is an object to save pq
    with tf.Session() as sess:
        # initialized the save op
        saver=tf.train.Saver()
        if adv:
            ckpt_restore_path = "Pretrain/%s/FM/%s/" % (dataset,time_stamp)
            ckpt_save_path = "Pretrain/%s/AFM/%s/" % (dataset, time_stamp)
        elif restore_time :
            ckpt_restore_path="Pretrain/%s/FM/%s/" % (dataset,restore_time)
            ckpt_save_path = "Pretrain/%s/FM/%s/" % (dataset, restore_time)
        else:
            print("train from scratch")
            ckpt_restore_path = "Pretrain/%s/FM/%s/" % (dataset, time_stamp)
            ckpt_save_path = "Pretrain/%s/FM/%s/" % (dataset, time_stamp)


        if not os.path.exists(ckpt_save_path):
            os.makedirs(ckpt_save_path)
        if not os.path.exists(ckpt_restore_path):
            os.makedirs(ckpt_restore_path)

        sess.run(tf.global_variables_initializer())

        if restore_time or adv:
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckpt_restore_path + 'checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("after restore")
        if dataset=="iPinYou":
            data = iPinYou()

            train_gen = data.batch_generator({"gen_type": "train", "batch_size": 30})
        # else ....
        batch_size=train_gen.batch_size

        #summary

        if adv:
            print("start adv training")
            for ep in range(epoch_start,epoch_end+1):
                total_loss=0
                n_batches=0
                for X,y in train_gen:
                    a,opt_loss_adv,b=sess.run([model.update_delta,model.opt_loss_adv,model.optimizer_adv],feed_dict={model.inputs:X,model.labels:y.reshape((batch_size,1))})
                    total_loss+=opt_loss_adv
                    n_batches+=1
                    print("n_batches",n_batches)
                    if n_batches==5:
                        break
                if ep %20==0:
                    saver.save(sess, ckpt_save_path, global_step=ep)
                print ("adv epoch {0}, average_loss {1}".format(ep,total_loss/(n_batches*batch_size)))
            print("adv training over")
        else:
            print("start training")
            for ep in range(epoch_start,epoch_end+1):
                total_loss=0
                n_batches=0
                for X,y in train_gen:
                    opt_loss_adv,a=sess.run([model.opt_loss,model.optimizer],feed_dict={model.inputs:X,model.labels:y.reshape(batch_size,1)})
                    total_loss+=opt_loss_adv
                    n_batches+=1
                    print("n_batches", n_batches)
                    if n_batches==5:
                        break
                if ep %20==0:
                    saver.save(sess, ckpt_save_path, global_step=ep)
                print ("epoch {0}, average_loss {1}".format(ep,total_loss/(n_batches*batch_size)))
            print ("training over")
        saver.save(sess, ckpt_save_path, global_step=epoch_end)

        print("test")
        test_gen=data.batch_generator({"gen_type":"test","batch_size":batch_size})
        total_log_loss = 0
        n_pos_preds=0
        n_batches = 0
        all_labels=[]
        all_preds=[]
        for X, y in test_gen:
            log_loss,preds = sess.run([model.log_loss,model.preds], feed_dict={model.inputs: X, model.labels: y.reshape(batch_size,1)})
            total_log_loss += log_loss
            n_batches += 1
            all_labels.extend(y)
            all_preds.extend(preds.flatten())
            if n_batches == 5:
                break
        auc=roc_auc_score(all_labels,all_preds)
        print( "auc {0},average_log_loss {1}".format("auc", total_log_loss/(n_batches * batch_size)))

if __name__ == '__main__':

    time_stamp = strftime('%Y_%m_%d_%H_%M_%S', localtime())
    model=FM()
    model.build_graph()
    training(model,"iPinYou",0,10,time_stamp)
    training(model,"iPinYou",11,20,time_stamp,adv=True)
    training(model,"iPinYou",11,20,adv=False,restore_time=time_stamp)