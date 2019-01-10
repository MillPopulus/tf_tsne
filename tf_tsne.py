import tensorflow as tf
import numpy as np
import os
# from sklearn.manifold._utils import _binary_search_perplexity

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True

class TSNE:
    def __init__(self,n_components=2, perplexity=30, early_exaggeration=12, learning_rate=500, n_iter=1000, momentum=0.8, verbose=0):
        self.learning_rate=learning_rate
        self.perplexity=perplexity
        self.n_components=n_components
        self.early_exaggeration=early_exaggeration
        self.n_iter=n_iter
        self.verbose=verbose
        self.momentum=momentum
    
    def fit_transform(self, P_coor):
        with tf.Graph().as_default():
            p2, p, sigma_mean, dists=TSNE.p_joint(P_coor, self.perplexity)
            sigma_mean=tf.Variable(sigma_mean, trainable=False)
            P_=tf.Variable(p2*self.early_exaggeration, trainable=False)
            P=tf.stop_gradient(P_)
            Q_coor=tf.Variable(tf.random_normal([tf.shape(P_coor)[0], self.n_components]))
            momentum=tf.Variable(0.8, trainable=False)
            Q_coor_loss, grad=TSNE.tsne(P, Q_coor)
            opt=TSNE.gradient_descent(Q_coor_loss, grad, Q_coor, self.learning_rate, momentum)
            grad_norm=tf.linalg.norm(grad)
        #         opt=tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=momentum)
        #         grad=opt.compute_gradients(Q_coor_loss, var_list=[Q_coor])
        #         grad_norm=tf.linalg.norm(tf.concat([x[0] for x in grad], axis=0))
        #         update_Q_coor=opt.apply_gradients(grad)
            with tf.Session(config=config) as sess:
                sess.run(tf.global_variables_initializer())                
                if self.verbose>=2:
                    print("sigma mean:", sess.run(sigma_mean))
                for i in range(self.n_iter):
        #                 if i is 20:
        #                     sess.run(momentum.assign(0.8))
                    if i is 100:
                        sess.run(P_.assign(P/self.early_exaggeration))
                        if self.verbose>=2 :
                            print("early exaggeration end.") # refering to sklearn
                    q, _, loss, gn= sess.run([Q_coor, opt, Q_coor_loss,grad_norm])
                    if self.verbose>=2 and i % 50 == 0:
                        print("Iteration {} loss: {}, grad norm: {:.6f}".format(i, loss, gn))
                return q       

    @staticmethod
    def remove_diag(x):
        diag_not_mask=~tf.cast(tf.diag(tf.ones(x.shape[:1])), dtype=tf.bool)
        with_out_diag=tf.reshape(tf.boolean_mask(x, diag_not_mask), x.shape-np.array([0,1]))
        return with_out_diag    
    
    @staticmethod
    def add_diag(x):
        xshape=tf.shape(x)
        tmp=tf.reshape(x, [xshape[1],xshape[0]])
        a=tf.zeros([xshape[1], 1], dtype=tf.float32)
        tmp=tf.concat([tmp,a], axis=1)
        tmp=tf.concat([[0.0], tf.reshape(tmp, [-1])], axis=0)
        tmp=tf.reshape(tmp, [xshape[0], xshape[0]])
        return tmp

    @staticmethod
    def squared_dists(x, diag=False):  # |x_i - x_j|^2
        sum_square=tf.reduce_sum(tf.square(x), axis=1, keepdims=True)
        dists=tf.maximum(sum_square -2*x@tf.transpose(x) +tf.transpose(sum_square),1e-6) # relu against negtive caused by overflow
        if diag:
            return dists
        else:    
            return TSNE.remove_diag(dists)

    @staticmethod
    def set_diag_zero(x):
        return tf.linalg.set_diag(x ,tf.zeros(tf.shape(x)[:1]))

    @staticmethod
    def cross_entropy(x, y, axis=-1):
        safe_y = tf.where(tf.equal(x, 0.), tf.ones_like(y), y)
        return -tf.reduce_sum(x * tf.log(safe_y), axis)

    @staticmethod
    def softmax_entropy_with_logits(logits, axis=-1): # H=-sum(p*log(p)) where p=softmax(logits)
        P=tf.nn.softmax(logits, axis=axis)
        H=tf.reduce_logsumexp(logits, axis=axis)- tf.reduce_sum(P*logits, axis=axis) # LSE(logits)-E(logits)
        '''
             -sum(p*log(p))
            =-sum(p*log_softmax(logits))
            =-sum(p*(logits-lse(logits)))
            =sum(p*lse(logits)-p*logits)
            =sum(p)*lse(logits)-sum(p*logits)
            =lse(logits)-E(logits)
        '''
        return H,P

    @staticmethod
    def calc_perplexity_and_probs(neg_dists, betas): # betas=1/2*sigmas^2
        logits=neg_dists*tf.reshape(betas,[-1,1])
        return TSNE.softmax_entropy_with_logits(logits)

    @staticmethod
    def binary_search_sigma(neg_dists, target, tol=1e-5, max_iter=50, lower=1e-20, upper=1000000.):
        #loop initial value
        target_entropy=np.log(target)

        def body(lows, ups, ans, finding_mask, x):
            finding_indices=tf.cast(tf.where(finding_mask), tf.int32)
            guess=(lows+ups)/2
            val2, _=TSNE.calc_perplexity_and_probs(tf.boolean_mask(neg_dists, finding_mask), tf.boolean_mask(guess, finding_mask)) # !TODO: compare the speed
            val=tf.scatter_nd(finding_indices, val2, tf.shape(finding_mask))
            diff=val-target_entropy
            new_ans_mask= ((tf.abs(diff)<= tol) | tf.equal(x+1, max_iter)) & finding_mask
            new_finding_mask= ~new_ans_mask & finding_mask
            greater_mask= (diff<- tol) & finding_mask
            leq_mask= (diff>tol) & finding_mask
#             dependencies=[
#                             tf.Print(val, [val], "val ",summarize=10),
#                             tf.Print(val2, [val2], "val2",summarize=10),
#                             tf.Print(guess, [guess], "guess:",summarize=10),
#                             tf.Print(greater_mask, [greater_mask], "gm ",summarize=10),
#                             tf.Print(ups, [ups], "ups ",summarize=10),
#                             tf.Print(leq_mask, [leq_mask], "lem ",summarize=10),
#                             tf.Print(lows, [lows], "lows",summarize=10),
#                             tf.Print(finding_mask, [finding_mask], "\nfm  ",summarize=10),
#                             tf.Print(ans, [ans], "ans",summarize=10),
#                             tf.Print(new_finding_mask, [new_finding_mask], "nfm ",summarize=10),
#                             tf.Print(new_ans_mask,[new_ans_mask], 'nam ',summarize=10),
#                             tf.Print(finding_indices,[finding_indices],'fid ',summarize=10),
#                             tf.print("x ", x)
#                          ]
#             with tf.control_dependencies(dependencies):
            return [tf.where(leq_mask, guess, lows),
                    tf.where(greater_mask, guess, ups),                        
                    tf.where(new_ans_mask, guess, ans),
                    new_finding_mask, 
                    tf.add(x,1)
                   ]

        cond= lambda a,b,ans,finding_mask,x: tf.reduce_any(finding_mask) & (x<max_iter)

        nums=tf.shape(neg_dists)[:1]
        lows=tf.fill(nums, lower)
        ups=tf.fill(nums, upper)
        finding_mask=tf.fill(nums, True)

        res=tf.while_loop(cond, body ,(lows, ups, lows, finding_mask,0), back_prop=False)
        ans=res[2]
        pra_iter=res[4]
#         with tf.control_dependencies([tf.Assert(pra_iter<max_iter, ['exceeded_max_iter'])]):
#         print("[Warning] exceeded mat iter, maybe sigma's precision is not enough.")
        return tf.identity(ans, name='betas')

#     @staticmethod
#     def transpose_without_diag(x):
#         xshape=tf.shape(x)
#         tmp=tf.reshape(x, xshape[::-1])
#         a=tf.zeros([xshape[1], 1], dtype=tf.float32)
#         tmp=tf.concat([tmp,a], axis=1)
#         tmp=tf.concat([[0.0], tf.reshape(tmp, [-1])], axis=0)
#         tmp=tf.reshape(tmp, [xshape[0], xshape[0]])
#         # origin got
#         tmp=tf.reshape(tf.transpose(tmp),[-1])[1:]
#         tmp=tf.reshape(tmp,[-1, xshape[0]+1])[:,:-1]
#         tmp=tf.reshape(tmp, xshape)
#         return tmp    

    @staticmethod
    def p_joint(x, target_perplexity):
        neg_dists_no_diag=-TSNE.squared_dists(x, diag=False)
        betas= TSNE.binary_search_sigma(neg_dists_no_diag, target_perplexity)
        p=tf.nn.softmax(neg_dists_no_diag*tf.reshape(betas, [-1,1]))
        p=TSNE.add_diag(p)
        p=p/tf.reduce_sum(p, axis=-1, keepdims=True)
        return (p+tf.transpose(p))/(2*tf.cast(tf.shape(x)[0], dtype=tf.float32)), p, tf.reduce_mean(tf.sqrt(1/betas)), neg_dists_no_diag        
        
#         sum_square=np.sum(np.square(x), axis=1, keepdims=True)
#         dists=np.maximum(sum_square -2*x@x.T + sum_square.T, 1e-6)
#         p= _binary_search_perplexity(dists, None, target_perplexity, 6)
#         p=(p+p.T)/(2*p.shape[0])  
#         return p.astype(np.float32), p, tf.constant([1]), dists
    
    @staticmethod
#     @tf.custom_gradient
    def tsne(p,y):
        dists=TSNE.squared_dists(y, diag=True)
        q_num=TSNE.set_diag_zero(1/(1+dists))
        q=tf.nn.relu(q_num/tf.reduce_sum(q_num))
        y=tf.expand_dims(y, axis=-2)
        y_cross_diff= y-tf.transpose(y, [1,0,2])
#         L2=tf.reduce_sum(dists) grddddd>>>??
        loss= -tf.reduce_sum(TSNE.cross_entropy(p,p)-TSNE.cross_entropy(p,q))
        grad= tf.reduce_sum((tf.expand_dims((p-q)*q_num, axis=-1))*y_cross_diff, axis=1)   
        return loss, grad
    
    def gradient_descent(loss, grad, x, lr, momentum, min_gain=0.01):
        gains=tf.Variable(tf.ones_like(x, dtype=tf.float32))
        update=tf.Variable(tf.zeros_like(x, dtype=tf.float32))
        direct= update*grad < 0.0
        gains=gains.assign(tf.maximum(tf.where(direct, gains+0.2, gains*0.8), min_gain))
        update=update.assign(update*momentum - lr*grad*gains)
        return x.assign(x+update)
