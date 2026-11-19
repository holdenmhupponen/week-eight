# Building a 'Stupid' Language Model
- loaded the github file into google colab - no errors
- installed the preliminary 'lego blocks' - no errors
- ran second cell - no errors
- tried to run third cell to set a text document variable - received the error "cat: '*.txt': No such file or directory"
  - after trying some things around, I figured that something must have gone wrong with the 2nd cell because that's where the file should have come from
  - so, I ran second cell again and, lo and behold, the third cell now worked without error
  - I'm not sure why it didn't work the first time even though no error message appeared and it showed me the green checkmark
- tried to run cell nine - received a large error with 6 frames:
  - "---------------------------------------------------------------------------
    
    ValueError                                Traceback (most recent call last)
    
    /tmp/ipykernel_711/1684944084.py in <cell line: 0>()
          1 sess = gpt2.start_tf_sess()
          2 
    ----> 3 gpt2.finetune(sess,
          4               dataset=file_name,
          5               model_name='124M',
    
    
    6 frames
    /usr/local/lib/python3.13/dist-packages/gpt_2_simple/gpt_2.py in finetune(sess, dataset, steps, model_name, model_dir, combine, batch_size, learning_rate, accumulate_gradients, restore_from, run_name, checkpoint_dir, sample_every, sample_length, sample_num, multi_gpu, save_every, print_every, max_checkpoints, use_memory_saving_gradients, only_train_transformer_layers, optimizer, overwrite, reuse)
        196         gpus = get_available_gpus()
        197 
    --> 198     output = model.model(hparams=hparams, X=context, gpus=gpus, reuse=reuse)
        199     loss = tf.reduce_mean(
        200         input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(
    
    
    /usr/local/lib/python3.13/dist-packages/gpt_2_simple/src/model.py in model(hparams, X, past, scope, gpus, reuse)
        186         batch, sequence = shape_list(X)
        187 
    --> 188         wpe = tf.compat.v1.get_variable('wpe', [hparams.n_ctx, hparams.n_embd],
        189                              initializer=tf.compat.v1.random_normal_initializer(stddev=0.01))
        190         wte = tf.compat.v1.get_variable('wte', [hparams.n_vocab, hparams.n_embd],
    
    
    /usr/local/lib/python3.13/dist-packages/tensorflow/python/ops/variable_scope.py in get_variable(name, shape, dtype, initializer, regularizer, trainable, collections, caching_device, partitioner, validate_shape, use_resource, custom_getter, constraint, synchronization, aggregation)
       1563                  synchronization=VariableSynchronization.AUTO,
       1564                  aggregation=VariableAggregation.NONE):
    -> 1565   return get_variable_scope().get_variable(
       1566       _get_default_variable_store(),
       1567       name,
    
    
    /usr/local/lib/python3.13/dist-packages/tensorflow/python/ops/variable_scope.py in get_variable(self, var_store, name, shape, dtype, initializer, regularizer, reuse, trainable, collections, caching_device, partitioner, validate_shape, use_resource, custom_getter, constraint, synchronization, aggregation)
       1273       if dtype is None:
       1274         dtype = self._dtype
    -> 1275       return var_store.get_variable(
       1276           full_name,
       1277           shape=shape,
    
    
    /usr/local/lib/python3.13/dist-packages/tensorflow/python/ops/variable_scope.py in get_variable(self, name, shape, dtype, initializer, regularizer, reuse, trainable, collections, caching_device, partitioner, validate_shape, use_resource, custom_getter, constraint, synchronization, aggregation)
        518       return custom_getter(**custom_getter_kwargs)
        519     else:
    --> 520       return _true_getter(
        521           name,
        522           shape=shape,
    
    
    /usr/local/lib/python3.13/dist-packages/tensorflow/python/ops/variable_scope.py in _true_getter(name, shape, dtype, initializer, regularizer, reuse, trainable, collections, caching_device, partitioner, validate_shape, use_resource, constraint, synchronization, aggregation)
        471             "name was already created with partitioning?" % name)
        472 
    --> 473       return self._get_single_variable(
        474           name=name,
        475           shape=shape,
    
    
    /usr/local/lib/python3.13/dist-packages/tensorflow/python/ops/variable_scope.py in _get_single_variable(self, name, shape, dtype, initializer, regularizer, partition_info, reuse, trainable, collections, caching_device, validate_shape, use_resource, constraint, synchronization, aggregation)
        829         # ResourceVariables don't have an op associated with so no traceback
        830         if isinstance(var, resource_variable_ops.ResourceVariable):
    --> 831           raise ValueError(err_msg)
        832         tb = var.op.traceback[::-1]
        833         # Throw away internal tf entries and only take a few lines. In some
    
    
    ValueError: Variable model/wpe already exists, disallowed. Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope?"
  - tried to rerun a bunch of things like last time, to no avail
  - since I have no programming experience, I clicked on the "Explain Errors" button
  - the Google Gemini feature added these two lines of code:
    - "import tensorflow as tf"
    - "tf.compat.v1.reset_default_graph()"
  - after running this, the cell worked without error
- ran eleventh cell - no error:
  - Local workers are being urged to remain vigilant while we perform maintenance.</p> <p>Follow Project Gutenberg on <a href="https://mastodon.social/@gutenberg_org">Mastodon</a>, <a href="https://bsky.app/profile/gutenberg.org">Bluesky</a>, or <a href="https://www.facebook.com/share/1A6Rcy9TLd/">Facebook</a>.</p> </main
    ====================
    Local workers are being asked to provide assistance while we perform maintenance.</p> <p>Follow Project Gutenberg on <a href="https://mastodon.social/@gutenberg_org">Mastodon</a>, <a href="https://bsky.app/profile/gutenberg.org">Bluesky</a>, or <a href="https://www.facebook.com/share/1A6Rcy9TLd/">Facebook</a>.</p> </main
    ====================
    Local workers are being urged to remain vigilant while we perform maintenance.</p> <p>Follow Project Gutenberg on <a href="https://mastodon.social/@gutenberg_org">Mastodon</a>, <a href="https://bsky.app/profile/gutenberg.org">Bluesky</a>, or <a href="https://www.facebook.com/share/1A6Rcy9TLd/">Facebook</a>.</p> </main
    ====================
    Local workers are temporarily unavailable while we perform maintenance.</p> <p>Follow Project Gutenberg on <a href="https://mastodon.social/@gutenberg_org">Mastodon</a>, <a href="https://bsky.app/profile/gutenberg.org">Bluesky</a>, or <a href="https://www.facebook.com/share/1A6Rcy9TLd/">Facebook</a>.</p> </main> </body
    ====================
    Local workers are temporarily unavailable while we perform maintenance.</p> <p>Follow Project Gutenberg on <a href="https://mastodon.social/@gutenberg_org">Mastodon</a>, <a href="https://bsky.app/profile/gutenberg.org">Bluesky</a>, or <a href="https://www.facebook.com/share/1A6Rcy9TLd/">Facebook</a>.</p> </main> </body
    ====================

# LLM-CLIP Archaeological Image Model
- setup notebook by installing llm transformer and llm-clip - no errors
- compressed my archaeological images, uploaded "Archaeological Images.zip" to colab - no errors
- tried to unzip the folder - one error:
  - since my file name had a space in it, it couldn't be read by the command
  - changed it to "Archaeological_Images.zip"; ran successfully
- edited the llm_clip.py file - no errors
- embedded the .jpg photos inside my folder - no errors
- ran the first query for 'pottery' and the second query - no errors
- made the display function - no errors
- queried for pottery and it showed other images like pottery - no errors
- queried the visually_similar function for images like mine labeled 'iron-nails.jpg', successfully showed me two others from my folder that looked pretty similar - no errors
