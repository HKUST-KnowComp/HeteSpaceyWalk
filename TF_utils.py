

import tensorflow as tf

# os.environ[“CUDA_DEVICE_ORDER”] = “PCI_BUS_ID” # 按照PCI_BUS_ID顺序从0开始排列GPU设备
# os.environ[“CUDA_VISIBLE_DEVICES”] = “0” #设置当前使用的GPU设备仅为0号设备  设备名称为'/gpu:0'
# os.environ[“CUDA_VISIBLE_DEVICES”] = “0,1” #设置当前使用的GPU设备为0,1号两个设备,名称依次为'/gpu:0'、'/gpu:1'
# os.environ[“CUDA_VISIBLE_DEVICES”] = “-1” #禁用GPU
# tf.device('/cpu:0') + tf.Session(config=tf.ConfigProto(device_count={'gpu':0})) #禁用GPU


def activation_summary(x):
    """ Helper to create summaries for activations.
        Creates a summary that provides a histogram of activations.
        Creates a summary that measures the sparsity of activations.

    Args:
        x: Tensor
    Returns:
        nothing
    """
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)  # Outputs a Summary protocol buffer with a histogram.
    tf.summary.scalar(tensor_name + '/sparsity',
                      tf.nn.zero_fraction(x))  # Outputs a Summary protocol buffer with scalar values.

def variable_with_weight_decay(name, initial_value, dtype = tf.float32, trainable = True, wd=None):
    """ Helper to create an initialized Variable with weight decay.
        Note that the Variable is initialized with a truncated normal distribution.
        A weight decay is added only if one is specified.

    Args:
        name: name of the variable
        initial_value: initial value for Variable
        wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.

    Returns:
        Variable Tensor
    """
    var = tf.Variable(initial_value=initial_value, name=name, trainable=trainable, dtype=dtype)
    if wd is not None and wd != 0:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name=name + '_loss')
        tf.summary.scalar(name + '_l2_loss', weight_decay)
        tf.add_to_collection('losses', weight_decay)
    return var



def add_loss_summaries(total_loss):
    """ Add summaries for losses.
        Generates moving average for all losses and associated summaries for visualizing the performance of the network.
        moving average -> eliminate noise

    Args:
        total_loss: Total loss from loss().
    Returns:
        loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    # The moving averages are computed using exponential decay:
    # shadow_variable -= (1 - decay) * (shadow_variable - variable)   equivalent to:
    # shadow_variable = decay * shadow_variable + (1 - decay) * variable
    loss_averages = tf.train.ExponentialMovingAverage(decay=0.999, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op



def train(total_loss, global_step, initial_learning_rate = 1, decay_steps = 0, decay_rate = 0.1):
    """ Create an optimizer and apply to all trainable variables.
            Add moving average for all trainable variables.

        Args:
            total_loss: total loss from loss().
            global_step: Integer Variable counting the number of training steps processed.

        Returns:
            train_op: op for training.
    """
    lr = initial_learning_rate
    if decay_steps > 0:
        # Decay the learning rate exponentially based on the number of steps.
        # decayed_learning_rate = initial_learning_rate * decay_rate ^ (global_step / decay_steps)
        lr = tf.train.exponential_decay(initial_learning_rate,
                                        global_step,
                                        decay_steps,
                                        decay_rate,
                                        staircase=True)

    tf.summary.scalar('learning_rate', lr)


    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = add_loss_summaries(total_loss)

    # Compute gradients
    with tf.control_dependencies([loss_averages_op]):
        # opt = tf.train.MomentumOptimizer(lr, graphcnn_option.MOMENTUM)
        # opt = tf.train.AdamOptimizer(lr)
        optimizer = tf.train.GradientDescentOptimizer(lr)
        train_op = optimizer.minimize(total_loss,
                                      global_step=global_step,
                                      gate_gradients=optimizer.GATE_NONE)

    # # Track the moving averages of all trainable variables.
    # variable_averages = tf.train.ExponentialMovingAverage(
    #     MOVING_AVERAGE_DECAY, global_step)
    # variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    #     train_op = tf.no_op(name='train')

    return train_op




