import tensorflow as tf
from lenet import LeNet
from initData import init
from testImages import load_images
# TODO: Fill this in based on where you saved the training and testing data

# load data
X_train_optimized, X_valid_optimized, X_test_optimized, y_train, y_valid, y_test = init()

### Train model.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))

one_hot_y = tf.one_hot(y, 43)

EPOCHS = 10
BATCH_SIZE = 32
rate = 0.00115

logits = LeNet(x)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)

loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)

training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train_optimized)

    print("Training...")
    for i in range(EPOCHS):
        X_train_optimized, y_train = shuffle(X_train_optimized, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train_optimized[offset:end], y_train[offset:end]
            summary = sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = evaluate(X_valid_optimized, y_valid)
        print("EPOCH {} ...".format(i + 1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('logs/2', sess.graph)

    saver.save(sess, './lenet')
    print("Model saved")


with tf.Session() as sess:
    saver.restore(sess, './lenet')

    sess.run(tf.global_variables_initializer())
    test_images = load_images()

    r = tf.argmax(logits, 1)

    predicted_logits = sess.run(logits, feed_dict={x: test_images})
    print(predicted_logits)