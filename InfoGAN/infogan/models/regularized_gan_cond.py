from infogan.misc.distributions import Product, Distribution, Gaussian, Categorical, Bernoulli
import prettytensor as pt
import tensorflow as tf
import infogan.misc.custom_ops
from infogan.misc.custom_ops import leaky_rectify
import pdb

class RegularizedGAN(object):
    def __init__(self, output_dist, latent_spec, batch_size, image_shape, network_type):
        """
        :type output_dist: Distribution
        :type latent_spec: list[(Distribution, bool)]
        :type batch_size: int
        :type network_type: string
        """
        self.output_dist = output_dist
        self.latent_spec = latent_spec
        self.latent_dist = Product([x for x, _ in latent_spec])
        self.reg_latent_dist = Product([x for x, reg in latent_spec if reg])
        self.nonreg_latent_dist = Product([x for x, reg in latent_spec if not reg])
        self.batch_size = batch_size
        self.network_type = network_type
        self.image_shape = image_shape
        assert all(isinstance(x, (Gaussian, Categorical, Bernoulli)) for x in self.reg_latent_dist.dists)

        self.reg_cont_latent_dist = Product([x for x in self.reg_latent_dist.dists if isinstance(x, Gaussian)])
        self.reg_disc_latent_dist = Product([x for x in self.reg_latent_dist.dists if isinstance(x, (Categorical, Bernoulli))])

        image_size = image_shape[0]
        if network_type == "mnist":
            with tf.variable_scope("d_net"):
                self.state_template = \
                     ((pt.template("state").
                     custom_fully_connected(128).
                     fc_batch_norm().
                     apply(tf.nn.elu)))

                shared_template = \
                    (pt.template("input").
                     #reshape([-1] + list(image_shape)).
                     #custom_conv2d(64, k_h=4, k_w=4).
                     custom_fully_connected(256).
                     apply(tf.nn.relu).
                     custom_fully_connected(512).
                     fc_batch_norm().
                     apply(tf.nn.sigmoid).
                     #dropout(0.55))
                     custom_fully_connected(34).
                     fc_batch_norm().
                     apply(leaky_rectify))
                self.discriminator_template = shared_template.custom_fully_connected(1)
                self.encoder_template = \
                    (shared_template.
                     custom_fully_connected(128).
                     fc_batch_norm().
                     apply(leaky_rectify).
                     custom_fully_connected(self.reg_latent_dist.dist_flat_dim))

            with tf.variable_scope("g_net"):
                self.generator_template = \
                    (pt.template("input").
                     custom_fully_connected(256).
                     fc_batch_norm().
                     apply(tf.nn.elu).
                     custom_fully_connected(128).
                     fc_batch_norm().
                     apply(tf.nn.elu).
                     custom_fully_connected(32).
                     fc_batch_norm().
                     apply(tf.nn.elu).
                     custom_fully_connected(6).
                     fc_batch_norm().
                     apply(tf.nn.sigmoid).
                     # custom_fully_connected(64).
                     # fc_batch_norm().
                     # apply(tf.nn.elu).
                     # custom_fully_connected(32).
                     # fc_batch_norm().
                     # dropout(0.75).
                     # apply(tf.nn.elu).
                     # custom_fully_connected(6).
                     # fc_batch_norm().
                     # apply(tf.nn.tanh).
                     #reshape([-1, int(image_size / 4), int(image_size / 4), 128]).
                     #custom_deconv2d([0, int(image_size / 2), int(image_size / 2), 64], k_h=4, k_w=4).
                     #fc_batch_norm().
                     #apply(tf.nn.relu).
                     #custom_deconv2d([0] + list(image_shape), k_h=4, k_w=4).
                     flatten())
        else:
            raise NotImplementedError

    def discriminate(self, x_var, state):
        state_out = self.state_template.construct(state = state)

        state_out = tf.convert_to_tensor(state_out, dtype=tf.float32)
        x_var = tf.convert_to_tensor(x_var, dtype= tf.float32)
        #whole_input = state_out.join([x_var], include_self=True)
        #whole_input = tf.convert_to_tensor(whole_input, dtype = tf.float32)
        whole_input = tf.concat([state_out, x_var],-1)

        d_out = self.discriminator_template.construct(input=whole_input)
        

        d = tf.nn.sigmoid(d_out[:, 0])
        
        #whole_input = tf.convert_to_tensor(whole_input, dtype = tf.float32)
        reg_dist_flat = self.encoder_template.construct(input=whole_input)


        reg_dist_info = self.reg_latent_dist.activate_dist(reg_dist_flat)

        #pdb.set_trace()
        return d, self.reg_latent_dist.sample(reg_dist_info), reg_dist_info, reg_dist_flat

    def generate(self, z_var, state):
        state_out = self.state_template.construct(state = state)
        z_var = tf.convert_to_tensor(z_var, dtype= tf.float32)

        
        whole_input = tf.concat([state_out, z_var],-1)
        #whole_input = state_out.join([z_var], include_self=True)
        x_dist_flat = self.generator_template.construct(input=whole_input)
        x_dist_info = self.output_dist.activate_dist(x_dist_flat)


        # return self.output_dist.sample(x_dist_info)*2 - 1, x_dist_info
        return x_dist_flat * (2) - 1, x_dist_info

    def disc_reg_z(self, reg_z_var):
        ret = []
        for dist_i, z_i in zip(self.reg_latent_dist.dists, self.reg_latent_dist.split_var(reg_z_var)):
            if isinstance(dist_i, (Categorical, Bernoulli)):
                ret.append(z_i)
        return self.reg_disc_latent_dist.join_vars(ret)

    def cont_reg_z(self, reg_z_var):
        ret = []
        for dist_i, z_i in zip(self.reg_latent_dist.dists, self.reg_latent_dist.split_var(reg_z_var)):
            if isinstance(dist_i, Gaussian):
                ret.append(z_i)
        return self.reg_cont_latent_dist.join_vars(ret)

    def disc_reg_dist_info(self, reg_dist_info):
        ret = []
        for dist_i, dist_info_i in zip(self.reg_latent_dist.dists, self.reg_latent_dist.split_dist_info(reg_dist_info)):
            if isinstance(dist_i, (Categorical, Bernoulli)):
                ret.append(dist_info_i)
        return self.reg_disc_latent_dist.join_dist_infos(ret)

    def cont_reg_dist_info(self, reg_dist_info):
        ret = []
        for dist_i, dist_info_i in zip(self.reg_latent_dist.dists, self.reg_latent_dist.split_dist_info(reg_dist_info)):
            if isinstance(dist_i, Gaussian):
                ret.append(dist_info_i)
        return self.reg_cont_latent_dist.join_dist_infos(ret)

    def reg_z(self, z_var):
        ret = []
        for (_, reg_i), z_i in zip(self.latent_spec, self.latent_dist.split_var(z_var)):
            if reg_i:
                ret.append(z_i)
        return self.reg_latent_dist.join_vars(ret)

    def nonreg_z(self, z_var):
        ret = []
        for (_, reg_i), z_i in zip(self.latent_spec, self.latent_dist.split_var(z_var)):
            if not reg_i:
                ret.append(z_i)
        return self.nonreg_latent_dist.join_vars(ret)

    def reg_dist_info(self, dist_info):
        ret = []
        for (_, reg_i), dist_info_i in zip(self.latent_spec, self.latent_dist.split_dist_info(dist_info)):
            if reg_i:
                ret.append(dist_info_i)
        return self.reg_latent_dist.join_dist_infos(ret)

    def nonreg_dist_info(self, dist_info):
        ret = []
        for (_, reg_i), dist_info_i in zip(self.latent_spec, self.latent_dist.split_dist_info(dist_info)):
            if not reg_i:
                ret.append(dist_info_i)
        return self.nonreg_latent_dist.join_dist_infos(ret)

    def combine_reg_nonreg_z(self, reg_z_var, nonreg_z_var):
        reg_z_vars = self.reg_latent_dist.split_var(reg_z_var)
        reg_idx = 0
        nonreg_z_vars = self.nonreg_latent_dist.split_var(nonreg_z_var)
        nonreg_idx = 0
        ret = []
        for idx, (dist_i, reg_i) in enumerate(self.latent_spec):
            if reg_i:
                ret.append(reg_z_vars[reg_idx])
                reg_idx += 1
            else:
                ret.append(nonreg_z_vars[nonreg_idx])
                nonreg_idx += 1
        return self.latent_dist.join_vars(ret)

    def combine_reg_nonreg_dist_info(self, reg_dist_info, nonreg_dist_info):
        reg_dist_infos = self.reg_latent_dist.split_dist_info(reg_dist_info)
        reg_idx = 0
        nonreg_dist_infos = self.nonreg_latent_dist.split_dist_info(nonreg_dist_info)
        nonreg_idx = 0
        ret = []
        for idx, (dist_i, reg_i) in enumerate(self.latent_spec):
            if reg_i:
                ret.append(reg_dist_infos[reg_idx])
                reg_idx += 1
            else:
                ret.append(nonreg_dist_infos[nonreg_idx])
                nonreg_idx += 1
        return self.latent_dist.join_dist_infos(ret)
