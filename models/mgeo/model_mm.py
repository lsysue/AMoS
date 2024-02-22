from functools import partial
from .xbert import BertConfig, BertModel, BertForMaskedLM, BertOnlyMLMHead
import random

import torch
from torch import nn
import torch.nn.functional as F

class MGeo(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 tokenizer = None,
                 config = None,     
                 ):
        super().__init__()
        
        self.tokenizer = tokenizer 

        bert_config = BertConfig.from_pretrained(text_encoder)
        bert_config.gis_embedding = 0
        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)      
        text_width = self.text_encoder.config.hidden_size

        gis_config = BertConfig.from_json_file(config['gis_bert_config'])
        self.gis_encoder = BertModel(gis_config, add_pooling_layer=False)
        for param in self.gis_encoder.parameters():
            param.requires_grad = False
        gis_width = gis_config.hidden_size

        self.gis2text = nn.Linear(gis_width, text_width)

    def forward(self, text, gis):
        # if random.random() < 0.5:
        #     return self.text_forward(text, gis)
        # else:
        #     return self.tg_forward(text, gis)

        gis_output = self.gis_encoder(input_ids = gis.input_ids,
                                       attention_mask = gis.attention_mask,
                                       token_type_ids = gis.token_type_ids,
                                       rel_type_ids = gis.rel_type_ids,
                                       absolute_position_ids = gis.absolute_position_ids,
                                       relative_position_ids = gis.relative_position_ids,
                                       return_dict = True,
                                       mode='text',
                                      )
        gis_embeds = gis_output.last_hidden_state
        gis_atts = gis.attention_mask

        embedding_output = self.text_encoder.embeddings(input_ids=text.input_ids)

        merge_embeds = torch.cat([embedding_output, self.gis2text(gis_embeds)], dim=1)
        merge_atts = torch.cat([text.attention_mask, gis.attention_mask], dim=-1)

        text_output = self.text_encoder(attention_mask = merge_atts, encoder_embeds = merge_embeds, return_dict = True, mode = 'text')

        text_embeds = text_output.last_hidden_state
        # text_atts = text_output.attention_mask

        tl = embedding_output.size(1)
        il = gis_embeds.size(1)

        txt_emb = text_embeds[:, :tl, :]
        gis_emb = text_embeds[:, tl:tl+il, :]
        
        return txt_emb, gis_emb

