python __main__.py \
--dataset_dir /share/chaofan/code/FlagEmbedding_update/data/MTEB \
--embedder_name_or_path BAAI/bge-large-en-v1.5 \
--reranker_name_or_path BAAI/bge-reranker-large \
--query_instruction_for_retrieval "Represent this sentence for searching relevant passages: " \
--use_fp16 True \
--devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
--cache_dir /share/shared_models \
--corpus_embd_save_dir /share/chaofan/code/FlagEmbedding_update/data/BEIR_passage_embds \
--reranker_max_length 512 \
--use_special_instructions True \
--languages eng \
--task_types Retrieval Reranking Summarization STS Clustering PairClassification Classification \
--tasks ArguAna

ClimateFEVER DBPedia FEVER FiQA2018 HotpotQA MSMARCO NFCorpus NQ QuoraRetrieval SCIDOCS SciFact Touche2020 TRECCOVID CQADupstackAndroidRetrieval CQADupstackEnglishRetrieval CQADupstackGamingRetrieval CQADupstackGisRetrieval CQADupstackMathematicaRetrieval CQADupstackPhysicsRetrieval CQADupstackProgrammersRetrieval CQADupstackStatsRetrieval CQADupstackTexRetrieval CQADupstackUnixRetrieval CQADupstackWebmastersRetrieval CQADupstackWebmastersRetrieval CQADupstackWordpressRetrieval SummEval BIOSSES SICK-R STS12 STS13 STS14 STS15 STS16 STS17 STS22 STSBenchmark ArxivClusteringP2P ArxivClusteringS2S BiorxivClusteringP2P BiorxivClusteringS2S MedrxivClusteringP2P MedrxivClusteringS2S RedditClustering RedditClusteringP2P StackExchangeClustering StackExchangeClusteringP2P TwentyNewsgroupsClustering AskUbuntuDupQuestions MindSmallReranking SciDocsRR StackOverflowDupQuestions SprintDuplicateQuestions TwitterSemEval2015 TwitterURLCorpus AmazonCounterfactualClassification AmazonPolarityClassification AmazonReviewsClassification Banking77Classification EmotionClassification ImdbClassification MassiveIntentClassification MassiveScenarioClassification MTOPDomainClassification MTOPIntentClassification ToxicConversationsClassification TweetSentimentExtractionClassification
