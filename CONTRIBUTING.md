Hey, I recently used this library to use the BAAI/bge-en-icl for embedding text data in OCEAN form. It kept crashing out when I tried to run it on my local GPU using the FlagEmbedding and it kept returning "killed"

I made a small add on with the accelerate library which enabled it to split the model into CPU and GPU when it overflows. Within this file FlagEmbedding/FlagEmbedding/flag_models.py 
