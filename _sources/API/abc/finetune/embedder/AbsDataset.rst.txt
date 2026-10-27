==========
AbsDataset
==========

AbsEmbedderTrainDataset
=======================

.. autoclass:: FlagEmbedding.abc.finetune.embedder.AbsEmbedderTrainDataset

Methods
-------

.. automethod:: FlagEmbedding.abc.finetune.embedder.AbsEmbedderTrainDataset._load_dataset

.. automethod:: FlagEmbedding.abc.finetune.embedder.AbsEmbedderTrainDataset._shuffle_text

AbsEmbedderCollator
===================

.. autoclass:: FlagEmbedding.abc.finetune.embedder.AbsEmbedderCollator

AbsEmbedderSameDatasetTrainDataset
==================================

.. autoclass:: FlagEmbedding.abc.finetune.embedder.AbsEmbedderSameDatasetTrainDataset

Methods
-------

.. automethod:: FlagEmbedding.abc.finetune.embedder.AbsEmbedderSameDatasetTrainDataset.refresh_epoch

.. automethod:: FlagEmbedding.abc.finetune.embedder.AbsEmbedderSameDatasetTrainDataset._load_dataset

.. automethod:: FlagEmbedding.abc.finetune.embedder.AbsEmbedderSameDatasetTrainDataset._get_file_batch_size

.. automethod:: FlagEmbedding.abc.finetune.embedder.AbsEmbedderSameDatasetTrainDataset._get_train_group_size

.. automethod:: FlagEmbedding.abc.finetune.embedder.AbsEmbedderSameDatasetTrainDataset._create_batch_data

AbsEmbedderSameDatasetCollator
==============================

.. autoclass:: FlagEmbedding.abc.finetune.embedder.AbsEmbedderSameDatasetCollator

EmbedderTrainerCallbackForDataRefresh
=====================================

.. autoclass:: FlagEmbedding.abc.finetune.embedder.EmbedderTrainerCallbackForDataRefresh

Methods
-------

.. automethod:: FlagEmbedding.abc.finetune.embedder.EmbedderTrainerCallbackForDataRefresh.on_epoch_end