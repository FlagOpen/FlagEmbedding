AbsReranker
===========

.. autoclass:: FlagEmbedding.abc.inference.AbsReranker

Methods
-------

.. automethod:: FlagEmbedding.abc.inference.AbsReranker.get_target_devices

.. automethod:: FlagEmbedding.abc.inference.AbsReranker.get_detailed_instruct

.. automethod:: FlagEmbedding.abc.inference.AbsReranker.get_detailed_inputs

.. automethod:: FlagEmbedding.abc.inference.AbsReranker.compute_score

.. automethod:: FlagEmbedding.abc.inference.AbsReranker.compute_score_single_gpu

.. automethod:: FlagEmbedding.abc.inference.AbsReranker.start_multi_process_pool

.. automethod:: FlagEmbedding.abc.inference.AbsReranker.encode_multi_process

.. automethod:: FlagEmbedding.abc.inference.AbsReranker._encode_multi_process_worker

.. automethod:: FlagEmbedding.abc.inference.AbsReranker.stop_multi_process_pool