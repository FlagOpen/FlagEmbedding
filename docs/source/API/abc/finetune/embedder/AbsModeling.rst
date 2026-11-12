===========
AbsModeling
===========

AbsEmbedderModel
================

.. autoclass:: FlagEmbedding.abc.finetune.embedder.AbsEmbedderModel

Methods
-------

.. automethod:: FlagEmbedding.abc.finetune.embedder.AbsEmbedderModel.encode

.. automethod:: FlagEmbedding.abc.finetune.embedder.AbsEmbedderModel.compute_loss

.. automethod:: FlagEmbedding.abc.finetune.embedder.AbsEmbedderModel.compute_score

.. automethod:: FlagEmbedding.abc.finetune.embedder.AbsEmbedderModel.save

.. automethod:: FlagEmbedding.abc.finetune.embedder.AbsEmbedderModel.get_local_score

.. automethod:: FlagEmbedding.abc.finetune.embedder.AbsEmbedderModel.compute_local_score

.. automethod:: FlagEmbedding.abc.finetune.embedder.AbsEmbedderModel.forward

.. automethod:: FlagEmbedding.abc.finetune.embedder.AbsEmbedderModel.distill_loss

.. automethod:: FlagEmbedding.abc.finetune.embedder.AbsEmbedderModel._compute_no_in_batch_neg_loss

.. automethod:: FlagEmbedding.abc.finetune.embedder.AbsEmbedderModel._compute_in_batch_neg_loss

.. automethod:: FlagEmbedding.abc.finetune.embedder.AbsEmbedderModel._compute_cross_device_neg_loss

.. automethod:: FlagEmbedding.abc.finetune.embedder.AbsEmbedderModel._dist_gather_tensor


EmbedderOutput
==============

.. autoclass:: FlagEmbedding.abc.finetune.embedder.EmbedderOutput