from transformers import AutoModel, AutoConfig, AutoModelForMaskedLM, AutoTokenizer
import torch



def initial_emb(model, output_dir):
    target_len = 8194

    position_ids = torch.arange(target_len, dtype=torch.long)
    position_ids = position_ids.unsqueeze(0)
    # create hierarchical embedding
    alpha = 0.4
    pos_ids = torch.arange(model.config.max_position_embeddings, dtype=torch.long)
    if hasattr(model, 'roberta'):
        position_embeddings = model.roberta.embeddings.position_embeddings(pos_ids)
        model.roberta.embeddings.position_ids = torch.arange(target_len).expand((1, -1))
    else:
        position_embeddings = model.embeddings.position_embeddings(pos_ids)
        model.embeddings.position_ids = torch.arange(target_len).expand((1, -1))

    position_embeddings = position_embeddings - alpha * position_embeddings[:1]
    position_embeddings = position_embeddings / (1-alpha)

    embedding_x = []
    embedding_y = []
    for i in range(position_ids.size(0)):
        pos_embedding_x = torch.index_select(position_embeddings, 0, position_ids[i, :] // model.config.max_position_embeddings)
        pos_embedding_y = torch.index_select(position_embeddings, 0, position_ids[i, :] % model.config.max_position_embeddings)
        embedding_x.append(pos_embedding_x.unsqueeze(0))
        embedding_y.append(pos_embedding_y.unsqueeze(0))

    pos_embedding_x = torch.cat(embedding_x, 0)
    pos_embedding_y = torch.cat(embedding_y, 0)


    position_embeddings = alpha * pos_embedding_x + (1-alpha) * pos_embedding_y
    position_embeddings = position_embeddings.squeeze(dim=0)

    if hasattr(model, 'roberta'):
        diff = torch.sum(torch.abs(position_embeddings[:model.config.max_position_embeddings] - model.roberta.embeddings.position_embeddings(pos_ids)), dim=-1)
    else:
        diff = torch.sum(torch.abs(position_embeddings[:model.config.max_position_embeddings] - model.embeddings.position_embeddings(pos_ids)), dim=-1)
    print(diff.size())
    print(diff)
    print(position_embeddings.size())

    model.config.max_position_embeddings = target_len
    embedding_new = torch.nn.Embedding(target_len, 1024)
    embedding_new.weight = torch.nn.Parameter(position_embeddings)
    if hasattr(model, 'roberta'):
        model.roberta.embeddings.position_embeddings = embedding_new
    else:
        model.embeddings.position_embeddings = embedding_new
    model.save_pretrained(output_dir)
    print(model.config)
    print(model)


model_name = 'xlm-roberta-large'
model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenzier = AutoTokenizer.from_pretrained(model_name)
print(tokenzier)
tokenzier.model_max_length=8192
initial_emb(model, output_dir='/share/models/xlm-roberta-large-8194')
tokenzier.save_pretrained('/share/models/xlm-roberta-large-8194')