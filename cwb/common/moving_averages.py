import tensorflow as tf

class MovingAverages:
    def __init__(self, namespace, var_list, ckpt_kwargs, decay=0.99, float_dtype=tf.float32):
        zero_initilizer = tf.zeros_initializer()
        self.namespace = namespace
        self.n_vars = len(var_list)
        self.var_list = var_list # HACK: assume this is consistent in order
        self.avg_var_list = [tf.Variable(initial_value=zero_initilizer(shape=var.shape, dtype=float_dtype), dtype=float_dtype) for var in self.var_list]
        for i in range(self.n_vars):
            ckpt_kwargs[self.get_kth_name(i)] = self.avg_var_list[i]
        self.decay = decay

    def get_kth_name(self, k):
        return '__avg/{}_shadow_{}'.format(self.namespace, k)

    def update_step(self):
        for i in range(self.n_vars):
            var = self.var_list[i]
            avg_var = self.avg_var_list[i]
            avg_var.assign_sub((1 - self.decay) * (avg_var - var))

    def swap_in_averages(self):
        self.backup_list = []
        for i in range(self.n_vars):
            self.backup_list.append(self.var_list[i].value())
            self.var_list[i].assign(self.avg_var_list[i].value())

    def swap_out_averages(self):
        for i in range(self.n_vars):
            self.var_list[i].assign(self.backup_list[i])
