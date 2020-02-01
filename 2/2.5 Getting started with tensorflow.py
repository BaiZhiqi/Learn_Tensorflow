import tensorflow as tf
import numpy as np
import input_data
import math

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("train_dir","MNIST_data/","The input file dir")
flags.DEFINE_bool("fake_data",False,"Whether one hot")
flags.DEFINE_float("learning_rate",1e-3,"Learning_rate")
flags.DEFINE_integer("max_steps",10000,"Max training steps")
flags.DEFINE_integer("batch_size",50,"Batch size")
def one_hot(labels,NUM_CLASSES):
    batch_size = tf.size(labels)
    labels = tf.expand_dims(labels, 1)
    indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
    concated = tf.concat([indices, labels],1)
    onehot_labels = tf.sparse_to_dense(
        concated, tf.stack([batch_size, NUM_CLASSES]), 1.0, 0.0)
    return onehot_labels


def initial_weight(name,shape):
    '''
    :param name: 每一层的名字
    :param shape: 每一层的输入尺寸
    :return: 返回初始化的权重
    '''
    # 每一层都创建于一个唯一的tf.name_scope之下，创建于该作用域之下的所有元素都将带有其前缀。
    # 通过tf.truncated_normal函数初始化权重变量，给赋予的shape则是一个二维tensor，其中第一个维度代表该层中权重变量所连接（connect from）的单元数量，
    # 第二个维度代表该层中权重变量所连接到的（connect to）单元数量。对于名叫hidden1的第一层，相应的维度则是[IMAGE_PIXELS, hidden1_units]，
    # 因为权重变量将图像输入连接到了hidden1层。tf.truncated_normal初始函数将根据所得到的均值和标准差，生成一个随机分布。
    with tf.name_scope(name) as scope:
        weights = tf.Variable(
            tf.truncated_normal(shape,
                                stddev=1.0 / math.sqrt(float(shape[0]))),
            name='weights')
        biases = tf.Variable(tf.zeros(shape[-1]),
                             name='biases')
    return weights, biases

def build_model():
    x = tf.placeholder(tf.float32,[None,784])
    y = tf.placeholder(tf.int32, [None])
    onehot_labels = one_hot(y, 10)
    #第一层
    weights, biases = initial_weight("hidden1",[784,200])
    hidden1 = tf.nn.relu(tf.matmul(x,weights)+biases)

    #第二层
    weights, biases = initial_weight("hidden1", [200, 100])
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

    #第三层
    weights, biases = initial_weight("hidden1", [100, 10])
    logits = tf.matmul(hidden2, weights) + biases

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                            labels=onehot_labels,
                                                            name='xentropy')
    loss = tf.reduce_mean(cross_entropy,name='xentropy_mean')

    correct_prediction = tf.equal(tf.argmax(onehot_labels, 1), tf.argmax(logits, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"),name="acc")
    #以下添加的属性会显示在tensorboard里，其中第一个参数为名字，第二个为要显示的变量
    tf.summary.scalar(loss.op.name, loss)
    tf.summary.scalar("acc", accuracy)
    optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)

    #global_step是全局训练步骤的数值
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)

    #训练
    data_sets = input_data.read_data_sets(FLAGS.train_dir, FLAGS.fake_data)
    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        # 为了释放TensorBoard所使用的事件文件（events file），
        # 所有的即时数据（在这里只有一个）都要在图表构建阶段合并至一个操作（op）中。
        summary_op = tf.summary.merge_all()
        # 用于写入包含了图表本身和即时数据具体值的事件文件。
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir,
                                             graph_def=sess.graph_def)
        #保存中间点
        saver = tf.train.Saver()

        #使用之前保存的权重进行继续训练
        #model_file = tf.train.latest_checkpoint('MNIST_data/')
        #saver.restore(sess, model_file)

        for step in range(FLAGS.max_steps):
            images_feed, labels_feed = data_sets.train.next_batch(FLAGS.batch_size)

            _, loss_value,acc = sess.run([train_op, loss,accuracy],feed_dict={x:images_feed,y:labels_feed})
            summary_str = sess.run(summary_op, feed_dict={x: images_feed, y: labels_feed})
            if step % 100 == 0:

                #保存中间点
                saver.save(sess, FLAGS.train_dir, global_step=step)
                summary_writer.add_summary(summary_str, step)
                print("Step {}: loss = {}, acc = {}".format(step,loss_value, acc))
        #测试
        print("Training Data Eval:{}".format(
            accuracy.eval(feed_dict={x:data_sets.train.images,y:data_sets.train.labels})))

        print("Validation Data Eval:{}".format(
            accuracy.eval(feed_dict={x: data_sets.validation.images, y: data_sets.validation.labels} )
        ))

        print("Test Data Eval:{}".format(
            accuracy.eval(feed_dict={x: data_sets.test.images, y: data_sets.test.labels} )
        ))

        #构建评估图标
        eval_correct = tf.nn.in_top_k(logits, y, 1)
        true_count = 0
        for step in range(200):
            images_feed, labels_feed = data_sets.test.next_batch(FLAGS.batch_size)
            true_count += sum(sess.run(eval_correct, feed_dict={x:images_feed,y:labels_feed}))
        total_num = 10000
        precision = float(true_count) / float(total_num)
        print(" Num examples: {}  Num correct: {} Precision:{}" .format(
            total_num, true_count, precision))
if __name__ == '__main__':
    build_model()