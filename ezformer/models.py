import tensorflow as tf
import tensorflow_hub as tf_hub


#Performer model code
class Performer:
    #TODO: Implement a class for Perform model, by @shirondru
    #https://github.com/shirondru/enformer_fine_tuning
    pass

#Enformer model code
class Enformer:

    def __init__(self, tfhub_url):
        self._model = tf_hub.load(tfhub_url).model

    def predict_on_batch(self, inputs):
        predictions = self._model.predict_on_batch(inputs)
        # return a dictionary of the predictions
        return {k: v.numpy() for k, v in predictions.items()}

    def contribution_input_grad(self, input_sequence, target_mask, output_head='human'):
        input_sequence = input_sequence[tf.newaxis]

        target_mask_mass = tf.reduce_sum(target_mask)
        with tf.GradientTape() as tape:
            tape.watch(input_sequence)
            prediction = tf.reduce_sum(
                target_mask[tf.newaxis] *
                self._model.predict_on_batch(input_sequence)[output_head]) / target_mask_mass

        input_grad = tape.gradient(prediction, input_sequence) * input_sequence
        input_grad = tf.squeeze(input_grad, axis=0)
        return tf.reduce_sum(input_grad, axis=-1)
