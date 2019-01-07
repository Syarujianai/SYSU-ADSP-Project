# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import layers
import input_processor

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
  "OUTPUT_DIR", "./data/model/",
  "the output directory where the model checkpints will be written"
)

flags.DEFINE_string(
  "TRAIN_ANNOTA_PATH", "./data/annotation.txt",
  "splited train set annotation"
)

flags.DEFINE_string(
  "EVAL_ANNOTA_PATH", "./data/val_annotation.txt",
  "splited val set annotation"
)

flags.DEFINE_integer(
  "TRAIN_BATCH_SIZE", 4,
  "batch size of training"
)

flags.DEFINE_integer(
  "EVAL_BATCH_SIZE", 1,
  "batch size of training"
)

flags.DEFINE_integer(
  "TRAIN_EPOCH", 10,
  "training epochs"
)

   
def cnn_model_fn(features, labels, mode):
  '''Model function for CNN Classifier.
  '''
  # Input Layer
  input_layer = tf.reshape(features, [-1, 129, 33, 1])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Dense Layer, [-1, 33 * 9 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 32 * 8 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer, units=2
  logits = tf.layers.dense(inputs=dropout, units=2)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  # Output tf.log in shell
  tf.logging.set_verbosity(tf.logging.INFO)
  
  # Load training and eval data
  train_input_fn, train_epoch_steps = input_processor.input_process_fn_builder(FLAGS.TRAIN_ANNOTA_PATH)
  eval_input_fn, eval_epoch_steps = input_processor.input_process_fn_builder(FLAGS.EVAL_ANNOTA_PATH)
  
  # Create the Estimator
  config = tf.estimator.RunConfig(
                save_checkpoints_steps=1000,
                log_step_count_steps=10,
                save_summary_steps=10,
                keep_checkpoint_max=10)
  classifier = tf.estimator.Estimator(
                model_fn=cnn_model_fn, model_dir=FLAGS.OUTPUT_DIR, config=config)

  # Set up logging for training
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
                tensors=tensors_to_log, every_n_iter=50)
                
  # Train and evaluate
  # lambda expression - refer: https://www.tensorflow.org/guide/premade_estimators#create_input_functions
  train_spec = tf.estimator.TrainSpec(
                input_fn=lambda: train_input_fn(FLAGS.TRAIN_ANNOTA_PATH, FLAGS.TRAIN_BATCH_SIZE),
                max_steps=train_epoch_steps*FLAGS.TRAIN_EPOCH,
                hooks=[logging_hook])
  eval_spec = tf.estimator.EvalSpec(
                input_fn=lambda: eval_input_fn(FLAGS.EVAL_ANNOTA_PATH, FLAGS.EVAL_BATCH_SIZE),
                steps=eval_epoch_steps*FLAGS.TRAIN_EPOCH,
                throttle_secs=900)
  tf.estimator.train_and_evaluate(
                classifier,
                train_spec,
                eval_spec
              )
              

if __name__ == "__main__":
  tf.app.run()
